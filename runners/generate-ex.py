"""
Extended FlowMDM Generation Script (generate-ex)

Non-intrusive extension of the original `generate.py` adding optional export of
SMPL(H) parameters (Babel dataset) and construction of a SMPLX-compatible pose
tensor, plus optional validation through a SMPL-X forward pass.

Default behaviour (when no new flags are used) mirrors the original script.

Available CLI Arguments (inherited from generate.py):
--------------------------------------------------------
Base Options:
  --cuda                   Use cuda device, otherwise use CPU (default: True)
  --device DEVICE          Device id to use (default: 0)
  --seed SEED              For fixing random seed (default: 10)
  --batch_size SIZE        Batch size during training (default: 64)

Sampling Options:
  --model_path PATH        Path to model####.pt file to be sampled (REQUIRED)
  --output_dir DIR         Path to results dir (auto created by the script)
                          If empty, will create dir in parallel to checkpoint
  --num_samples N          Maximal number of prompts to sample (default: 10)
                          If loading dataset from file, this field will be ignored
  --num_repetitions N      Number of repetitions per sample (default: 3)
  --guidance_param S       For classifier-free sampling - specifies the s parameter (default: 2.5)

Frame Sampler Options (Babel dataset):
  --min_seq_len LENGTH     Babel dataset FrameSampler minimum length (default: 70)
  --max_seq_len LENGTH     Babel dataset FrameSampler maximum length (default: 200)

Generation Options:
  --instructions_file PATH Path to a json file with instructions for sequences generation
                          If empty, will take text prompts from dataset
                          JSON format: {"text": [...], "lengths": [...]}
  --split SPLIT           Split to be used for generation: train, val, test (default: test)
  --sample_gt             Sample and visualize gt instead of generate sample

Unfolding Options:
  --transition_length N    For evaluation - take margin around transition (default: 60)

Model Options (auto-loaded from model's args.json, can be overridden):
  --dataset DATASET        Dataset name: humanml, babel (auto-loaded)
  --bpe_denoising_step N   Denoising step for APE→RPE transition (default: 100)
                          0 for all RPE, -1 or >=diffusion_steps for all APE
  --use_chunked_att        Use chunked windowed local/relative attention (LongFormer-style)
  --unconstrained          Model is trained unconditionally (no text/action conditioning)

Added CLI Arguments (new in generate-ex):
------------------------------------------
  --export-smpl            Export raw SMPLH parameters per sample (Babel only)
  --export-smplx           Also build zero-padded SMPLX axis-angle pose
  --smplx-model-path PATH  Path to PARENT directory containing 'smplx' folder with model files
                          (e.g., use "../../data" NOT "../../data/smplx")
                          If provided & smplx installed, validates poses via forward pass

Additional Outputs (when export flags enabled):
    smpl_params.npy              list[dict] raw SMPLH params (allow_pickle)
            smplx_pose.npy               list[np.ndarray] SMPLX-aligned pose arrays (axis-angle; non-zero via SciPy / NumPy fallback)
    smplx_transl.npy             list[np.ndarray] per-frame root translations (added for locomotion playback)
    smplx_global_orient.npy      list[np.ndarray] per-frame global orientation axis-angle (T,3) per sample (convenience)
    smplx_global_orient_mat.npy  list[np.ndarray] per-frame global rotation matrices (T,3,3) per sample
    smplx_root_transform.npy     list[np.ndarray] per-frame 4x4 root transforms (T,4,4) in original SMPL-X (Z-up) coordinates
    smplx_layout.json            Layout, shapes, metadata
    smplx_vertices_preview.npy   (optional) small vertex subset for quick check

Usage Examples:
--------------
# Basic generation (same as generate.py)
python -m runners.generate-ex --model_path ./results/babel/FlowMDM/model001300000.pt \\
    --instructions_file ./tests/simple-walk/simple_walk_instructions.json \\
    --num_repetitions 1 --bpe_denoising_step 125 --guidance_param 1.5 --dataset babel

# With SMPL export and validation (note: ../../data contains the 'smplx' folder)
python -m runners.generate-ex --model_path ./results/babel/FlowMDM/model001300000.pt \\
    --instructions_file ./tests/simple-walk/simple_walk_instructions.json \\
    --export-smpl --export-smplx --smplx-model-path ../../data \\
    --output_dir ./my_output

# From dataset samples with custom seed
python -m runners.generate-ex --model_path ./results/babel/FlowMDM/model001300000.pt \\
    --num_samples 5 --seed 42 --export-smplx

References:
  - howto-export-smpl-params-and-convert-to-smplx.md (explain folder)
"""
import json
import os

import numpy as np
import torch

from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import load_model
from utils.parser_util import generate_args as base_generate_args
from diffusion.diffusion_wrappers import DiffusionWrapper_FlowMDM as DiffusionWrapper
from data_loaders.get_data import get_dataset_loader
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion_mix
from runners.generate import feats_to_xyz  # reuse tested conversion

datasets_fps = {"humanml": 20, "babel": 30}


def extend_args():
    import sys
    import argparse
    
    # First, parse our custom arguments from sys.argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--export-smpl', action='store_true', help='Export raw SMPL(H) parameters (Babel only).')
    parser.add_argument('--export-smplx', action='store_true', help='Also export SMPLX-aligned pose (pads missing components).')
    parser.add_argument('--smplx-model-path', type=str, default=None, help='Optional SMPL-X model path for validation forward pass.')
    
    # Parse known args to extract our custom ones
    custom_args, remaining = parser.parse_known_args()
    
    # Temporarily modify sys.argv to only contain the base arguments
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining
    
    # Get the base args using the original parser
    args = base_generate_args()
    
    # Restore original sys.argv
    sys.argv = original_argv
    
    # Add our custom args to the base args
    args.export_smpl = custom_args.export_smpl
    args.export_smplx = custom_args.export_smplx
    args.smplx_model_path = custom_args.smplx_model_path
    
    return args


def matrix_to_axis_angle(matrix):
    """Deprecated helper (kept for compatibility). Not used; pytorch3d removed."""
    raise RuntimeError("matrix_to_axis_angle helper is deprecated (pytorch3d removed)")

def extract_smpl_params(datastruct):
    """Extract SMPL parameters from SlimSMPLDatastruct."""
    # Get rotation and translation data
    rots_data = datastruct.rots
    rots = rots_data.rots  # Shape: [T, num_joints, 3, 3] - rotation matrices
    trans = rots_data.trans  # Shape: [T, 3] - translations
    
    # Convert to numpy if tensor
    if hasattr(rots, 'cpu'):
        rots = rots.cpu().numpy()
    if hasattr(trans, 'cpu'):
        trans = trans.cpu().numpy()
    
    # Extract components (following SMPLH structure)
    global_orient_mat = rots[:, 0]  # [T, 3, 3]
    body_pose_mat = rots[:, 1:22]  # [T, 21, 3, 3]
    
    # Check if hands are present
    has_hands = rots.shape[1] > 22
    if has_hands:
        hand_pose_mat = rots[:, 22:]  # [T, 30, 3, 3]
        left_hand_pose_mat = hand_pose_mat[:, :15]  # [T, 15, 3, 3]
        right_hand_pose_mat = hand_pose_mat[:, 15:]  # [T, 15, 3, 3]
    else:
        left_hand_pose_mat = None
        right_hand_pose_mat = None
    
    # Convert matrices to axis-angle - SciPy first, NumPy Rodrigues fallback
    try:
        from scipy.spatial.transform import Rotation

        def mat2aa_scipy(mat):
            if mat is None:
                return None
            shape = mat.shape
            flat = mat.reshape(-1, 3, 3)
            aa = Rotation.from_matrix(flat).as_rotvec().astype(np.float32)
            return aa.reshape(shape[:-2] + (3,))

        global_orient = mat2aa_scipy(global_orient_mat)
        body_pose = mat2aa_scipy(body_pose_mat).reshape(body_pose_mat.shape[0], -1)
        left_hand_pose = mat2aa_scipy(left_hand_pose_mat).reshape(left_hand_pose_mat.shape[0], -1) if left_hand_pose_mat is not None else None
        right_hand_pose = mat2aa_scipy(right_hand_pose_mat).reshape(right_hand_pose_mat.shape[0], -1) if right_hand_pose_mat is not None else None
        print("[info] Using scipy.spatial.transform.Rotation for matrix->axis-angle conversion")
    except Exception:
        def rodrigues_numpy(mat):
            if mat is None:
                return None
            flat = mat.reshape(-1, 3, 3)
            aa_list = []
            for R in flat:
                R = R.astype(np.float64)
                trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
                theta = np.arccos(trace)
                if theta < 1e-8:
                    aa_list.append(np.zeros(3, dtype=np.float32))
                    continue
                if np.pi - theta < 1e-4:
                    w = np.array([
                        np.sqrt(max(0, (R[0, 0] + 1) / 2.0)),
                        np.sqrt(max(0, (R[1, 1] + 1) / 2.0)),
                        np.sqrt(max(0, (R[2, 2] + 1) / 2.0)),
                    ])
                    w[0] = np.copysign(w[0], R[2, 1] - R[1, 2])
                    w[1] = np.copysign(w[1], R[0, 2] - R[2, 0])
                    w[2] = np.copysign(w[2], R[1, 0] - R[0, 1])
                    axis = w / (np.linalg.norm(w) + 1e-8)
                else:
                    axis = np.array([
                        R[2, 1] - R[1, 2],
                        R[0, 2] - R[2, 0],
                        R[1, 0] - R[0, 1],
                    ]) / (2 * np.sin(theta))
                aa_list.append((axis * theta).astype(np.float32))
            aa = np.stack(aa_list, axis=0)
            return aa.reshape(mat.shape[:-2] + (3,))

        global_orient = rodrigues_numpy(global_orient_mat)
        body_pose = rodrigues_numpy(body_pose_mat).reshape(body_pose_mat.shape[0], -1)
        left_hand_pose = rodrigues_numpy(left_hand_pose_mat).reshape(left_hand_pose_mat.shape[0], -1) if left_hand_pose_mat is not None else None
        right_hand_pose = rodrigues_numpy(right_hand_pose_mat).reshape(right_hand_pose_mat.shape[0], -1) if right_hand_pose_mat is not None else None
        print("[info] Using pure NumPy Rodrigues fallback for matrix->axis-angle conversion (SciPy missing)")

    # Debug ranges (first few frames) to ensure not all zeros
    try:
        print(f"       global_orient frame0: {global_orient[0]}")
        print(f"       body_pose norm frame0: {np.linalg.norm(body_pose[0]):.4f}")
    except Exception:
        pass
    
    # Default betas (shape parameters) - assumed constant
    betas = np.zeros(10, dtype=np.float32)
    
    return {
        'global_orient': global_orient,
        'body_pose': body_pose,
        'left_hand_pose': left_hand_pose,
        'right_hand_pose': right_hand_pose,
        'transl': trans,
        'betas': betas
    }

def build_smplx_pose(smpl_params):
    """Build SMPLX-compatible pose array from SMPL parameters."""
    global_orient = smpl_params['global_orient']  # [T, 3]
    body_pose = smpl_params['body_pose']  # [T, 63]
    left_hand = smpl_params.get('left_hand_pose')  # [T, 45] or None
    right_hand = smpl_params.get('right_hand_pose')  # [T, 45] or None
    
    T = global_orient.shape[0]
    
    # Build pose array
    parts = [global_orient, body_pose]
    
    # Add hand poses or zeros
    if left_hand is not None and right_hand is not None:
        parts.extend([left_hand, right_hand])
        has_hands = True
    else:
        # Pad with zeros for hands
        parts.extend([np.zeros((T, 45), dtype=np.float32), 
                     np.zeros((T, 45), dtype=np.float32)])
        has_hands = False
    
    # Pad jaw, leye, reye (SMPLX-specific)
    parts.extend([
        np.zeros((T, 3), dtype=np.float32),  # jaw
        np.zeros((T, 3), dtype=np.float32),  # leye  
        np.zeros((T, 3), dtype=np.float32)   # reye
    ])
    
    # Concatenate all parts
    pose = np.concatenate(parts, axis=1)  # [T, 165]
    
    layout = ['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 
              'jaw_pose', 'leye_pose', 'reye_pose']
    
    return pose, layout, {'has_hands': has_hands, 'dims': pose.shape[1]}


def axis_angle_to_matrix_batch(aa):
    """Convert axis-angle (T,3) to rotation matrix (T,3,3).

    Uses SciPy when available, otherwise a NumPy Rodrigues implementation.
    """
    if aa is None:
        return None
    try:
        from scipy.spatial.transform import Rotation
        mats = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().astype(np.float32)
        return mats.reshape(aa.shape[0], 3, 3)
    except Exception:
        # Fallback pure numpy
        mats = []
        for v in aa:
            theta = np.linalg.norm(v)
            if theta < 1e-8:
                mats.append(np.eye(3, dtype=np.float32))
                continue
            k = v / theta
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float32)
            R = np.eye(3, dtype=np.float32) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
            mats.append(R.astype(np.float32))
        return np.stack(mats, axis=0)


def build_root_transforms(global_orient_aa, transl):
    """Build per-frame 4x4 root transforms (T,4,4) from global orient axis-angle and translation.

    All in original SMPL-X coordinate system (typically Z-up)."""
    Rmats = axis_angle_to_matrix_batch(global_orient_aa)
    T = global_orient_aa.shape[0]
    transforms = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    transforms[:, :3, :3] = Rmats
    if transl is not None:
        transforms[:, :3, 3] = transl
    return Rmats, transforms

def feats_to_xyz_with_smpl(sample, dataset, batch_size=1):
    """
    Modified version of feats_to_xyz that also returns SMPL parameters for Babel.
    """
    if dataset == 'babel':
        from data_loaders.amass.transforms import SlimSMPLTransform
        transform = SlimSMPLTransform(batch_size=batch_size, name='SlimSMPLTransform', ename='smplnh', normalization=True)
        all_feature = sample  # [bs, nfeats, 1, seq_len]
        all_feature_squeeze = all_feature.squeeze(2)  # [bs, nfeats, seq_len]
        all_feature_permutes = all_feature_squeeze.permute(0, 2, 1)  # [bs, seq_len, nfeats]
        splitted = torch.split(all_feature_permutes, all_feature.shape[0])  # list of [seq_len, nfeats]
        
        sample_list = []
        smpl_params_list = []
        
        for seq in splitted[0]:
            all_features = seq
            Datastruct = transform.SlimDatastruct
            datastruct = Datastruct(features=all_features)
            
            # Get joints
            sample = datastruct.joints
            sample_list.append(sample.permute(1, 2, 0).unsqueeze(0))
            
            # Extract SMPL parameters
            smpl_params = extract_smpl_params(datastruct)
            smpl_params_list.append(smpl_params)
        
        joints = torch.cat(sample_list)
        return joints, smpl_params_list
    else:
        # For non-Babel datasets, just return joints without SMPL params
        joints = feats_to_xyz(sample, dataset, batch_size=batch_size)
        return joints, []

def validate_with_smplx(smpl_params, model_path, gender='neutral'):
    """Validate SMPL parameters using SMPLX model forward pass."""
    try:
        import smplx
        import torch
        
        # Adjust model path - user passes parent dir, SMPLX expects parent/smplx/
        # The create function will append 'smplx' to the path internally
        
        T = smpl_params['global_orient'].shape[0]
        
        # Create SMPLX model
        print(f"[info] Loading SMPLX model from {model_path} (will look in {model_path}/smplx/)")
        model = smplx.create(
            model_path=model_path,
            model_type='smplx',
            gender=gender,
            batch_size=T,
            use_pca=False,
            flat_hand_mean=False
        )
        
        # Prepare inputs
        global_orient = torch.from_numpy(smpl_params['global_orient']).float()  # [T, 3]
        body_pose = torch.from_numpy(smpl_params['body_pose']).float()  # [T, 63]
        transl = torch.from_numpy(smpl_params['transl']).float() if smpl_params['transl'] is not None else None
        betas = torch.from_numpy(smpl_params['betas']).float().unsqueeze(0).expand(T, -1)  # [T, 10]
        
        # Reshape body pose
        body_pose = body_pose.reshape(T, 21, 3)
        
        # Prepare hand poses
        if smpl_params['left_hand_pose'] is not None:
            left_hand_pose = torch.from_numpy(smpl_params['left_hand_pose']).float().reshape(T, 15, 3)
            right_hand_pose = torch.from_numpy(smpl_params['right_hand_pose']).float().reshape(T, 15, 3)
        else:
            left_hand_pose = torch.zeros(T, 15, 3)
            right_hand_pose = torch.zeros(T, 15, 3)
        
        # Forward pass
        output = model(
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=transl,
            betas=betas,
            jaw_pose=torch.zeros(T, 3),
            leye_pose=torch.zeros(T, 3),
            reye_pose=torch.zeros(T, 3),
            expression=torch.zeros(T, 10)
        )
        
        # Return validation info
        return {
            'success': True,
            'vertices_shape': output.vertices.shape,
            'joints_shape': output.joints.shape,
            'vertices_preview': output.vertices[:5].detach().cpu().numpy()  # First 5 frames
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def main():
    args = extend_args()
    
    # Early validation of smplx-model-path if provided
    if args.smplx_model_path:
        import os.path as osp
        smplx_subdir = osp.join(args.smplx_model_path, 'smplx')
        if not osp.exists(smplx_subdir):
            raise ValueError(
                f"[error] SMPLX model directory not found at {smplx_subdir}\n"
                f"        You provided: {args.smplx_model_path}\n"
                f"        This should be the PARENT directory containing 'smplx' folder.\n"
                f"        Example: use '../../data' instead of '../../data/smplx'"
            )
        
        # Check for model files
        expected_files = ['SMPLX_NEUTRAL.pkl', 'SMPLX_MALE.pkl', 'SMPLX_FEMALE.pkl']
        missing_files = [f for f in expected_files if not osp.exists(osp.join(smplx_subdir, f)) and not osp.exists(osp.join(smplx_subdir, f.replace('.pkl', '.npz')))]
        if missing_files:
            print(f"[warn] Some SMPLX model files may be missing: {missing_files}")
        else:
            print(f"[info] SMPLX model path validated: {args.smplx_model_path}/smplx/")
    
    fixseed(args.seed)
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    out_path = args.output_dir
    fps = datasets_fps[args.dataset]
    dist_util.setup_dist(args.device)
    if out_path == '':
        # Use same output directory naming as original generate.py
        out_path = os.path.join(os.path.dirname(args.model_path), 
                                '{}_s{}'.format(niter, args.seed))
        if args.instructions_file != '':
            out_path += '_' + os.path.basename(args.instructions_file).replace('.json', '').replace(' ', '_').replace('.', '')
    os.makedirs(out_path, exist_ok=True)

    # Input mode handling (same logic as original, condensed)
    is_using_data = args.instructions_file == ''
    if not is_using_data:
        with open(args.instructions_file) as f:
            instructions = json.load(f)
        num_instructions = len(instructions['text'])
        args.batch_size = num_instructions
        args.num_samples = 1
    else:
        num_instructions = args.num_samples
        args.batch_size = num_instructions
        args.num_samples = 1

    instructions = None  # ensure symbol always defined
    if is_using_data:
        if args.split == 'test' and args.dataset == 'babel':
            args.split = 'val'
        data = load_dataset(args, args.split)
        iterator = iter(data)
        sample_gt, model_kwargs = next(iterator)
        # save text file listing prompts
        j = {"sequence": []}
        for i in range(num_instructions):
            length = model_kwargs['y']['lengths'][i].item()
            text = model_kwargs['y']['text'][i]
            j['sequence'].append([length, text])
        with open(os.path.join(out_path, 'prompted_texts.json'), 'w') as f:
            json.dump(j, f)
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
    else:  # instructions mode
        with open(args.instructions_file) as f:
            instructions = json.load(f)
        json_lengths = instructions['lengths']
        json_texts = instructions['text']
        mask = torch.ones((len(json_texts), max(json_lengths)))
        for i, length in enumerate(json_lengths):
            mask[i, length:] = 0
        model_kwargs = {'y': {'mask': mask, 'lengths': torch.tensor(json_lengths), 'text': list(json_texts), 'tokens': ['']}}
        with open(os.path.join(out_path, 'prompted_texts.json'), 'w') as f:
            json.dump(instructions, f)
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

    print(list(zip(list(model_kwargs['y']['text']), list(model_kwargs['y']['lengths'].cpu().numpy()))))

    # Model + diffusion
    model, diffusion = load_model(args, dist_util.dev())
    diffusion = DiffusionWrapper(args, diffusion, model)

    all_motions, all_lengths, all_text = [], [], []

    # For SMPL parameter extraction if enabled
    all_smpl_params = [] if (args.dataset == 'babel' and args.export_smpl) else None
    
    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetition #{rep_i}]')
        sample = diffusion.p_sample_loop(clip_denoised=False, model_kwargs=model_kwargs, progress=True)
        
        # Extract SMPL params for Babel if export is enabled
        if args.dataset == 'babel' and args.export_smpl:
            joints, smpl_params_list = feats_to_xyz_with_smpl(sample, args.dataset, batch_size=args.batch_size)
            all_smpl_params.extend(smpl_params_list)
        else:
            joints = feats_to_xyz(sample, args.dataset, batch_size=args.batch_size)
        
        c_text = "".join([t + " /// " for t in model_kwargs['y']['text']])
        all_text.append(c_text)
        all_motions.append(joints.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].sum().unsqueeze(0))

    all_motions = np.concatenate(all_motions, axis=0)
    all_lengths = np.concatenate(all_lengths, axis=0)

    # Save standard results
    npy_path = os.path.join(out_path, 'results.npy')
    # Save dict with allow_pickle=True safely
    np.save(npy_path, np.array({'motion': all_motions,
                                'text': all_text,
                                'lengths': all_lengths,
                                'num_samples': args.num_samples,
                                'num_repetitions': args.num_repetitions}, dtype=object), allow_pickle=True)
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))

    # Optional exports
    if args.dataset == 'babel' and (args.export_smplx or args.export_smpl):
        if all_smpl_params:
            # Save raw SMPL parameters
            if args.export_smpl:
                np.save(os.path.join(out_path, 'smpl_params.npy'), np.array(all_smpl_params, dtype=object), allow_pickle=True)
                print(f'[info] Saved SMPL parameters for {len(all_smpl_params)} samples to smpl_params.npy')
            
            # Build and save SMPLX-compatible pose
            if args.export_smplx:
                smplx_poses = []
                for params in all_smpl_params:
                    pose, layout, meta = build_smplx_pose(params)
                    smplx_poses.append(pose)
                
                np.save(os.path.join(out_path, 'smplx_pose.npy'), np.array(smplx_poses, dtype=object), allow_pickle=True)
                # Also save translations for easier downstream use (list[T,3])
                smplx_transl = [p['transl'] for p in all_smpl_params]
                np.save(os.path.join(out_path, 'smplx_transl.npy'), np.array(smplx_transl, dtype=object), allow_pickle=True)
                # Save global orientations (axis-angle) explicitly for convenience
                smplx_global_orient = [p['global_orient'] for p in all_smpl_params]
                np.save(os.path.join(out_path, 'smplx_global_orient.npy'), np.array(smplx_global_orient, dtype=object), allow_pickle=True)
                # Build and save rotation matrices & 4x4 transforms
                global_orient_mats = []
                root_transforms = []
                for p in all_smpl_params:
                    mats, tfms = build_root_transforms(p['global_orient'], p['transl'])
                    global_orient_mats.append(mats)
                    root_transforms.append(tfms)
                np.save(os.path.join(out_path, 'smplx_global_orient_mat.npy'), np.array(global_orient_mats, dtype=object), allow_pickle=True)
                np.save(os.path.join(out_path, 'smplx_root_transform.npy'), np.array(root_transforms, dtype=object), allow_pickle=True)
                with open(os.path.join(out_path, 'smplx_layout.json'), 'w') as f:
                    json.dump({
                        'layout': layout,
                        'meta': meta,
                        'num_samples': len(smplx_poses),
                        'files': {
                            'pose': 'smplx_pose.npy',
                            'transl': 'smplx_transl.npy',
                            'global_orient': 'smplx_global_orient.npy',
                            'global_orient_mat': 'smplx_global_orient_mat.npy',
                            'root_transform': 'smplx_root_transform.npy'
                        },
                        'coordinate_system': 'SMPL-X original (Z-up)',
                        'notes': 'Root transforms are 4x4 matrices combining global orientation & translation in Z-up. For Y-up viewers rotate -90deg about X.'
                    }, f, indent=2)
                print('[info] Saved SMPLX-compatible poses (+ transl, global orient, matrices, root transforms)')
                
                # Optional validation with SMPLX model
                if args.smplx_model_path:
                    # Path was already validated at the start of main()
                    print(f'[info] Running SMPLX validation with model from {args.smplx_model_path}')
                    validation = validate_with_smplx(all_smpl_params[0], args.smplx_model_path)
                    
                    if validation['success']:
                        print('[info] SMPLX validation successful!')
                        print(f'       Vertices shape: {validation["vertices_shape"]}')
                        print(f'       Joints shape: {validation["joints_shape"]}')
                        # Save preview vertices
                        np.save(os.path.join(out_path, 'smplx_vertices_preview.npy'), 
                               validation['vertices_preview'])
                    else:
                        print(f'[warn] SMPLX validation failed: {validation["error"]}')
                        print('[hint] Make sure to pass the parent directory containing "smplx" folder')
                        print('[hint] e.g., use "../../data" instead of "../../data/smplx"')
        else:
            print('[warn] No SMPL parameters extracted. Make sure --export-smpl flag is set.')
    elif args.dataset != 'babel' and (args.export_smpl or args.export_smplx):
        print('[info] SMPL export flags ignored: dataset has no SMPL params.')

    # Visualization (same as original but simplified for brevity)
    skeleton = paramUtil.t2m_kinematic_chain
    from runners.generate import construct_template_variables  # reuse
    sample_print_template, row_print_template, sample_file_template, row_file_template = construct_template_variables(args.unconstrained)
    rep_files = []
    caption = ''  # ensure defined for save_multiple_samples call
    for rep_i in range(args.num_repetitions):
        caption = all_text[rep_i * args.num_samples]
        motion = all_motions[rep_i * args.num_samples].transpose(2, 0, 1)
        save_file = sample_file_template.format(rep_i)
        animation_save_path = os.path.join(out_path, save_file)
        lengths_list = model_kwargs['y']['lengths']
        captions_list = []
        for c, frame_len in zip(caption.split(' /// '), lengths_list):
            captions_list += [c, ] * frame_len
        try:
            plot_3d_motion_mix(animation_save_path, skeleton, motion, dataset=args.dataset, title=captions_list, fps=fps, vis_mode='alternate', lengths=lengths_list)
        except Exception as e:
            print(f'[warn] Visualization failed for rep {rep_i}: {e}')
        rep_files.append(animation_save_path)

    from runners.generate import save_multiple_samples
    save_multiple_samples(args, out_path, row_print_template, row_file_template, caption, rep_files)
    print(f'[Done] Extended results at {os.path.abspath(out_path)}')


def load_dataset(args, split):
    # Copied from original generate.py for isolation
    n_frames = 150
    if args.dataset == 'babel':
        args.num_frames = (args.min_seq_len, args.max_seq_len)
    else:
        args.num_frames = n_frames
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, split=split, load_mode='gen', protocol=args.protocol, pose_rep=args.pose_rep, num_workers=1)
    data.fixed_length = n_frames
    return data


if __name__ == '__main__':
    main()
