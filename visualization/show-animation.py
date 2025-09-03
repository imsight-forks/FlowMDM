"""FlowMDM Motion Animation Utility (PyVista / Qt)

Interactive visualization of generated FlowMDM text-to-motion (T2M) results.
Provides smooth real-time playback of 22‑joint skeleton sequences with
efficient in-place point updates (no per-frame allocations).

Key Features:
* Interactive 3D animation (play/pause, frame stepping, reset, quit)
* Auto-discovery of `results.npy` when a directory is provided
* Supports Babel (30 FPS) and HumanML3D (20 FPS) style outputs
* Efficient VTK text actor & label updates (no actor recreation)
* In‑place skeleton point updates via `np.copyto` for performance

Usage Examples:
    # Directory containing results.npy
    python visualization/show-animation.py results/babel/FlowMDM/001300000_s10_simple_walk_instructions

    # Direct file path
    python visualization/show-animation.py results/babel/FlowMDM/001300000_s10_simple_walk_instructions/results.npy

    # Autoplay and custom FPS
    python visualization/show-animation.py <RESULT_DIR> --autoplay --fps 24

Arguments:
    results_path   Path to a directory containing `results.npy` or the file itself
    --autoplay      Start playback immediately (default: start paused)
    --fps INT       Target playback FPS (default 30; adjust for slow/fast review)

Controls (inside the window):
    Space           Play / Pause
    Left / Right    Step one frame backward / forward (pauses playback)
    r               Reset to frame 0
    q               Quit / close window

Data Format:
    Motion array shape: (batch, 22, 3, seq_len)
    22 joints follow the T2M skeleton ordering used in HumanML3D / Babel.

Performance Notes:
    * Single PolyData for skeleton lines; points mutated in place each frame
    * Label anchor points updated in place; status text uses a persistent vtkTextActor
    * Timer callback interval derived from requested FPS (ms = 1000 / fps)

Recommended Environment:
    Run under the modern pixi "latest" env for newer PyVista & Qt stack:
        pixi run -e latest python visualization/show-animation.py <RESULT_DIR>
"""

import contextlib
import pathlib
import argparse

import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
import vtk

# Motion data loaded later from CLI argument (populated in __main__)
motion_data: np.ndarray | None = None
result: dict | None = None
flowmdm_result_file: pathlib.Path | None = None
input_path: pathlib.Path | None = None

# T2M joint indices and names (HumanML3D format)
t2m_joint_names: dict[int, str] = {
    0: "Pelvis",
    1: "L.Hip", 2: "R.Hip", 3: "Spine1",
    4: "L.Knee", 5: "R.Knee", 6: "Spine2",
    7: "L.Ankle", 8: "R.Ankle", 9: "Spine3",
    10: "L.Foot", 11: "R.Foot", 12: "Neck",
    13: "L.Collar", 14: "R.Collar", 15: "Head",
    16: "L.Shoulder", 17: "R.Shoulder",
    18: "L.Elbow", 19: "R.Elbow",
    20: "L.Wrist", 21: "R.Wrist"
}

# T2M kinematic chain structure from data_loaders/humanml/utils/paramUtil.py
t2m_kinematic_chain: list[list[int]] = [
    [0, 2, 5, 8, 11],      # Right leg: Pelvis → R.Hip → R.Knee → R.Ankle → R.Foot
    [0, 1, 4, 7, 10],      # Left leg: Pelvis → L.Hip → L.Knee → L.Ankle → L.Foot
    [0, 3, 6, 9, 12, 15],  # Spine: Pelvis → Spine1 → Spine2 → Spine3 → Neck → Head
    [9, 14, 17, 19, 21],   # Right arm: Spine3 → R.Collar → R.Shoulder → R.Elbow → R.Wrist
    [9, 13, 16, 18, 20]    # Left arm: Spine3 → L.Collar → L.Shoulder → L.Elbow → L.Wrist
]
# Build skeleton pairs once (list of (start,end) indices for lines)
skeleton_pairs: list[tuple[int, int]] = []
for _chain in t2m_kinematic_chain:
    for _i in range(len(_chain) - 1):
        skeleton_pairs.append((_chain[_i], _chain[_i + 1]))


class FlowMDMAnimator:
    """Interactive animator for FlowMDM motion data with in-place geometry updates.
    
    This class provides real-time 3D visualization of T2M format motion sequences
    with efficient geometry updates and interactive controls. The animation system
    uses in-place point updates to achieve smooth 30 FPS playback without memory
    allocation during the animation loop.
    
    Parameters
    ----------
    motion_data : np.ndarray
        Motion sequence data with shape (batch, 22, 3, seq_len) where:
        - batch: batch dimension (typically 1)
        - 22: number of T2M joints
        - 3: spatial coordinates (x, y, z)
        - seq_len: number of frames in the sequence
    
    Attributes
    ----------
    motion_data : np.ndarray
        The input motion sequence data
    current_frame : int
        Current animation frame index
    total_frames : int
        Total number of frames in the sequence
    is_playing : bool
        Animation playback state
    fps : float
        Target frames per second (30.0 for Babel dataset)
    plotter : pvqt.BackgroundPlotter
        PyVista Qt plotter for 3D rendering
    skel_poly : pv.PolyData
        Skeleton polydata with updateable point positions
    skel_actor : pv.Actor
        Skeleton mesh actor in the scene
    label_poly : pv.PolyData
        Joint label anchor points
    labels_actor : pv.Actor
        Joint label text actor
    vtk_text_actor : vtk.vtkTextActor
        Status text actor for efficient text updates
    key_joint_ids : list[int]
        Indices of joints to display labels for
    
    Methods
    -------
    render_frame(frame_index)
        Update geometry and render a specific frame
    toggle_animation()
        Toggle between play and pause states
    step_frame(direction)
        Step animation by one frame in given direction
    reset_animation()
        Reset animation to frame 0
    quit_animation()
        Stop animation and close window
    show()
        Start the interactive animation display
    """
    def __init__(self, motion_data: np.ndarray) -> None:
        """Initialize the FlowMDM animator with motion data.
        
        Sets up the 3D scene, skeleton geometry, text actors, and animation controls.
        Creates all necessary visualization components and prepares for real-time
        animation playback.
        
        Parameters
        ----------
        motion_data : np.ndarray
            Motion sequence data with shape (batch, 22, 3, seq_len)
            
        Notes
        -----
        This method performs several initialization steps:
        1. Creates PyVista Qt plotter window
        2. Sets up 3D scene with ground plane and axes
        3. Builds skeleton geometry from T2M kinematic chains
        4. Creates joint labels for key anatomical points
        5. Sets up efficient VTK text actor for status display
        6. Registers keyboard controls and timer callbacks
        """
        self.motion_data = motion_data
        self.current_frame = 0
        self.total_frames = motion_data.shape[-1]
        self.is_playing = False
        self.fps = 30.0

        # Plotter (Qt background)
        self.plotter = pvqt.BackgroundPlotter(  # type: ignore[attr-defined]
            title="FlowMDM T2M Motion Animation - 22 Joints",
            window_size=(1024, 768)
        )
        self._setup_scene()
        self._setup_controls()

        # Construct skeleton polydata once
        self.skel_poly = self._build_skeleton_polydata(0)
        self.skel_actor = self.plotter.add_mesh(  # type: ignore[attr-defined]
            self.skel_poly,
            color=[0.2, 0.4, 0.8],
            line_width=3.0,
            render_lines_as_tubes=True,
            point_size=10,
            render_points_as_spheres=True
        )

        # Labels - Show all important keypoints with larger font
        self.key_joint_ids = [0, 3, 6, 9, 12, 15, 1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21]  # All major joints
        first = self.motion_data[0, :, :, 0]
        self.label_poly = pv.PolyData(first[self.key_joint_ids].copy())
        self.labels_actor = self.plotter.add_point_labels(  # type: ignore[attr-defined]
            self.label_poly,
            [t2m_joint_names[i] for i in self.key_joint_ids],
            show_points=False,
            font_size=15,  # Increased from 10 to 15 (1.5x larger)
            always_visible=True
        )

        # Status text actor - Create VTK text actor directly for guaranteed efficient updates
        self.vtk_text_actor = vtk.vtkTextActor()
        self.vtk_text_actor.SetInput("Frame 0")
        self.vtk_text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.vtk_text_actor.GetPositionCoordinate().SetValue(0.02, 0.02)  # lower_left position
        
        # Set text properties - 1.5x larger font
        text_prop = self.vtk_text_actor.GetTextProperty()
        text_prop.SetFontSize(16)  # Increased from 11 to 16 (1.5x larger)
        text_prop.SetColor(0, 0, 0)
        text_prop.SetJustificationToLeft()
        text_prop.SetVerticalJustificationToBottom()
        
        # Add to plotter as VTK actor
        self.plotter.add_actor(self.vtk_text_actor, name='status_text')  # type: ignore[attr-defined]

        # Initial frame
        self.render_frame(0)

        # Register timer callback (approx every frame)
        self.plotter.add_callback(self._on_timer, interval=int(1000 / self.fps))  # type: ignore[attr-defined]

    # --- Setup helpers ---
    def _setup_scene(self) -> None:
        """Set up the 3D scene with ground plane, axes, and camera.
        
        Creates the basic 3D environment for motion visualization including:
        - Semi-transparent ground plane for spatial reference
        - 3D coordinate axes for orientation
        - Light gray background for better contrast
        - Optimal camera position for viewing human motion
        
        Notes
        -----
        The ground plane is positioned at y=0 (assuming y-up coordinate system)
        with grid lines for scale reference. Camera is positioned at [3,2,3]
        looking at origin with y-axis as up vector.
        """
        plane = pv.Plane(center=[0, 0, 0], direction=[0, 1, 0], i_size=4, j_size=4, i_resolution=20, j_resolution=20)
        self.plotter.add_mesh(plane, color=[0.6, 0.6, 0.6], opacity=0.3, show_edges=True, line_width=0.5)  # type: ignore[attr-defined]
        self.plotter.add_axes()  # type: ignore[attr-defined]
        self.plotter.set_background([0.95, 0.95, 0.95])  # type: ignore[attr-defined]
        self.plotter.camera_position = ([3, 2, 3], [0, 1, 0], [0, 1, 0])  # type: ignore[attr-defined]

    def _setup_controls(self) -> None:
        """Register keyboard event handlers for animation control.
        
        Sets up interactive keyboard controls for the animation:
        - Spacebar: Toggle play/pause
        - Left/Right arrows: Step frame by frame
        - 'r': Reset to frame 0
        - 'q': Quit application
        
        Notes
        -----
        Both ' ' (space) and 'space' key events are registered for
        play/pause functionality to ensure compatibility across
        different PyVista versions and platforms.
        """
        for key in (' ', 'space'):
            self.plotter.add_key_event(key, self.toggle_animation)  # type: ignore[attr-defined]
        self.plotter.add_key_event('Left', lambda: self.step_frame(-1))  # type: ignore[attr-defined]
        self.plotter.add_key_event('Right', lambda: self.step_frame(1))  # type: ignore[attr-defined]
        self.plotter.add_key_event('r', self.reset_animation)  # type: ignore[attr-defined]
        self.plotter.add_key_event('q', self.quit_animation)  # type: ignore[attr-defined]

    # --- Geometry helpers ---
    def _build_skeleton_polydata(self, frame_index: int) -> pv.PolyData:
        """Build PyVista PolyData for the skeleton at a specific frame.
        
        Creates a PolyData object containing the 22 joint positions and
        kinematic chain connections for the T2M skeleton format. The
        resulting geometry can be efficiently updated by modifying point
        positions without recreating the topology.
        
        Parameters
        ----------
        frame_index : int
            Frame number to extract joint positions from (0-based index)
            
        Returns
        -------
        pv.PolyData
            PolyData object with:
            - points: 22 joint positions (shape: 22x3)
            - lines: kinematic chain connections as line cells
            
        Notes
        -----
        The kinematic chain structure follows the T2M format:
        - Right leg: Pelvis → R.Hip → R.Knee → R.Ankle → R.Foot
        - Left leg: Pelvis → L.Hip → L.Knee → L.Ankle → L.Foot  
        - Spine: Pelvis → Spine1 → Spine2 → Spine3 → Neck → Head
        - Right arm: Spine3 → R.Collar → R.Shoulder → R.Elbow → R.Wrist
        - Left arm: Spine3 → L.Collar → L.Shoulder → L.Elbow → L.Wrist
        
        Line cells are formatted as [2, start_idx, end_idx] for each bone.
        """
        joints = self.motion_data[0, :, :, frame_index].copy()
        line_cells: list[int] = []
        for a, b in skeleton_pairs:
            line_cells.extend([2, a, b])
        poly = pv.PolyData()
        poly.points = joints
        poly.lines = np.array(line_cells)
        return poly

    # --- Frame / animation logic ---
    def render_frame(self, frame_index: int) -> None:
        """Render a specific animation frame with efficient geometry updates.
        
        Updates the 3D visualization to display the skeleton pose at the
        specified frame. Uses in-place geometry updates for maximum performance:
        - Updates skeleton joint positions via np.copyto()
        - Updates joint label positions
        - Updates status text with current frame info
        - Triggers scene re-rendering
        
        Parameters
        ----------
        frame_index : int
            Target frame to render (0-based index). Must be within
            [0, total_frames-1] range.
            
        Notes
        -----
        This method is optimized for real-time animation:
        - No new memory allocation during updates
        - Direct VTK text actor manipulation for status text
        - In-place point array updates using np.copyto()
        - Exception handling for label actor updates
        
        The status text displays: "F {frame}/{total-1} | {Play/Pause}"
        """
        if not (0 <= frame_index < self.total_frames):
            return
        self.current_frame = frame_index
        joints = self.motion_data[0, :, :, frame_index]
        # Update skeleton points
        pts = self.skel_poly.points
        np.copyto(pts, joints)
        self.skel_poly.points = pts
        # Update label anchors
        lpts = self.label_poly.points
        np.copyto(lpts, joints[self.key_joint_ids])
        self.label_poly.points = lpts
        with contextlib.suppress(Exception):
            self.labels_actor.SetInputData(self.label_poly)  # type: ignore[attr-defined]
        # Update status text - Direct VTK update (most efficient, no new actors)
        status = f"F {frame_index}/{self.total_frames-1} | {'Play' if self.is_playing else 'Pause'}"
        self.vtk_text_actor.SetInput(status)  # Direct VTK call - guaranteed efficient
        self.plotter.render()  # type: ignore[attr-defined]

    def _on_timer(self) -> None:
        """Timer callback for automatic animation playback.
        
        Called periodically (every ~33ms for 30 FPS) during animation playback.
        Advances to the next frame when playing, with automatic looping when
        reaching the end of the sequence.
        
        Notes
        -----
        Only advances frames when is_playing is True. Uses modulo arithmetic
        for seamless looping: (current_frame + 1) % total_frames.
        Timer interval is set during initialization: int(1000 / self.fps).
        """
        if self.is_playing:
            self.render_frame((self.current_frame + 1) % self.total_frames)

    # --- Control methods ---
    def toggle_animation(self) -> None:
        """Toggle between play and pause states.
        
        Switches the animation between playing and paused states.
        Prints the current state to console for user feedback.
        Bound to spacebar key events in the interactive window.
        """
        self.is_playing = not self.is_playing
        print("Playing" if self.is_playing else "Paused")

    def step_frame(self, direction: int) -> None:
        """Step animation by one frame in the specified direction.
        
        Advances or retreats the animation by exactly one frame.
        Automatically pauses playback and uses modulo arithmetic
        for seamless looping in both directions.
        
        Parameters
        ----------
        direction : int
            Direction to step: +1 for forward, -1 for backward
            
        Notes
        -----
        Bound to Left/Right arrow keys. Always pauses animation
        to allow precise frame-by-frame inspection.
        """
        self.is_playing = False
        self.render_frame((self.current_frame + direction) % self.total_frames)

    def reset_animation(self) -> None:
        """Reset animation to the first frame.
        
        Pauses playback and jumps to frame 0. Useful for restarting
        the animation sequence. Prints confirmation to console.
        Bound to 'r' key event in the interactive window.
        """
        self.is_playing = False
        self.render_frame(0)
        print("Animation reset")

    def quit_animation(self) -> None:
        """Stop animation and close the visualization window.
        
        Pauses playback and closes the PyVista plotter window,
        effectively terminating the application. Bound to 'q'
        key event for quick exit.
        """
        self.is_playing = False
        self.plotter.close()  # type: ignore[attr-defined]

    # --- UI ---
    def show(self) -> None:
        """Start the interactive animation display.
        
        Prints control instructions to console and opens the PyVista
        Qt window for interactive 3D visualization. This method blocks
        until the window is closed.
        
        Notes
        -----
        The animation starts in paused state at frame 0. Use spacebar
        to begin playback. The window remains responsive to all registered
        keyboard controls during display.
        
        Control summary printed to console:
        - Spacebar: Play/Pause animation
        - Left/Right arrows: Step frame by frame  
        - r: Reset to frame 0
        - q: Quit application
        """
        print("\nAnimation Controls:")
        print("  Spacebar: Play/Pause")
        print("  Left/Right arrows: Step frame")
        print("  r: Reset")
        print("  q: Quit")
        print("\nStarting interactive animation...")
        self.plotter.show()  # type: ignore[attr-defined]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive FlowMDM motion animation (PyVista)")
    parser.add_argument("results_path", help="Path to results directory or results.npy file (absolute or relative)")
    parser.add_argument("--autoplay", action="store_true", help="Start playing immediately instead of paused")
    parser.add_argument("--fps", type=int, default=30, help="Target playback FPS (default 30)")
    args = parser.parse_args()

    input_path = pathlib.Path(args.results_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Provided path does not exist: {input_path}")

    if input_path.is_dir():
        candidate = input_path / "results.npy"
        if not candidate.exists():
            found = list(input_path.glob("**/results.npy"))
            if not found:
                raise FileNotFoundError(f"Could not find results.npy inside directory: {input_path}")
            flowmdm_result_file = found[0]
        else:
            flowmdm_result_file = candidate
    else:
        flowmdm_result_file = input_path
        if flowmdm_result_file.name != "results.npy":
            raise ValueError("Results file must be named results.npy when providing a file path")

    result = np.load(flowmdm_result_file, allow_pickle=True).item()
    motion_data = result['motion']  # type: ignore[index]
    assert motion_data is not None, "Failed to load motion data"
    assert result is not None, "Failed to load results dict"

    print(f"Loaded motion data: {motion_data.shape}")
    print(f"Text prompts: {result.get('text', ['No text available'])}")
    print(f"Sequence lengths: {result.get('lengths', 'Not specified')}")
    print("FPS: 30 (Babel dataset)")
    print(f"Total frames: {motion_data.shape[-1]}")

    animator = FlowMDMAnimator(motion_data)
    animator.fps = float(args.fps)
    if args.autoplay:
        animator.toggle_animation()

    animator.show()

    app = getattr(animator.plotter, 'app', None)
    if app is not None:
        try:  # type: ignore[attr-defined]
            closing = getattr(app, 'closingDown', lambda: False)()
            if not closing:
                app.exec_()  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover
            print(f"[warn] Qt event loop failed: {e}")
