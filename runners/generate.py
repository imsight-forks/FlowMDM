# This code is based on https://github.com/openai/guided-diffusion
"""
FlowMDM Text-to-Motion Generation Script

Generates human motion compositions from text descriptions using FlowMDM's
diffusion-based approach with Blended Positional Encodings (BPE).

Usage Examples:
--------------
# Generate with instruction file
python -m runners.generate \
    --model_path ./results/humanml/FlowMDM/model000500000.pt \
    --instructions_file ./runners/jsons/composition_humanml.json \
    --bpe_denoising_step 60 --guidance_param 2.5

# Generate from dataset samples  
python -m runners.generate \
    --model_path ./results/babel/FlowMDM/model001300000.pt \
    --num_samples 5 --num_repetitions 3 \
    --bpe_denoising_step 125 --guidance_param 1.5

Key Parameters:
--------------
- model_path: Path to pretrained FlowMDM model (.pt file)
- instructions_file: JSON file with text descriptions and frame lengths
- bpe_denoising_step: BPE transition point (lower=smoother, higher=better quality per action)  
- guidance_param: Text conditioning strength (0=unconditioned, 2.5=strong conditioning)
- use_chunked_att: Memory optimization for long sequences (recommended for HumanML3D)
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import load_model
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion_mix
import json
from diffusion.diffusion_wrappers import DiffusionWrapper_FlowMDM as DiffusionWrapper

datasets_fps = {
    "humanml": 20,
    "babel": 30
}

def feats_to_xyz(sample, dataset, batch_size=1):
    """
    Convert motion features to 3D joint coordinates for visualization.
    
    Transforms the motion representation from the model's feature space to 
    3D joint positions that can be visualized as skeletal animations.

    Parameters
    ----------
    sample : torch.Tensor
        Motion features from the diffusion model. Shape varies by dataset:
        - HumanML3D: [batch_size, n_feats, 1, seq_len] where n_feats=263
        - Babel: [batch_size, 135, 1, seq_len] (SMPL parameters)
    dataset : str
        Dataset name. Must be either 'humanml' or 'babel'.
        Determines the feature format and conversion method.
    batch_size : int, optional
        Batch size for processing, by default 1.

    Returns
    -------
    torch.Tensor
        3D joint coordinates with shape [batch_size, n_joints, 3, seq_len]
        where n_joints=22 for both datasets, representing skeletal structure.

    Raises
    ------
    NotImplementedError
        If dataset is not 'humanml' or 'babel'.

    Notes
    -----
    - HumanML3D uses rotation-invariant coordinates (RIC) representation
    - Babel uses SMPL parameters that are converted via SlimSMPLTransform
    - The output follows T2M kinematic chain convention for visualization
    
    Examples
    --------
    >>> # Convert HumanML3D features to 3D coordinates
    >>> motion_3d = feats_to_xyz(sample, 'humanml')
    >>> motion_3d.shape  # [1, 22, 3, seq_len]
    """
    if dataset == 'humanml': # for HumanML3D
        n_joints = 22
        mean = np.load('dataset/HML_Mean_Gen.npy')
        std = np.load('dataset/HML_Std_Gen.npy')
        sample = sample.cpu().permute(0, 2, 3, 1)
        sample = (sample * std + mean).float()
        sample = recover_from_ric(sample, n_joints) # --> [1, 1, seqlen, njoints, 3]
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1) # --> [1, njoints, 3, seqlen]
    elif dataset == 'babel': # [bs, 135, 1, seq_len] --> 6 * 22 + 3 for trajectory
        from data_loaders.amass.transforms import SlimSMPLTransform
        transform = SlimSMPLTransform(batch_size=batch_size, name='SlimSMPLTransform', ename='smplnh', normalization=True)
        all_feature = sample #[bs, nfeats, 1, seq_len]
        all_feature_squeeze = all_feature.squeeze(2) #[bs, nfeats, seq_len]
        all_feature_permutes = all_feature_squeeze.permute(0, 2, 1) #[bs, seq_len, nfeats]
        splitted = torch.split(all_feature_permutes, all_feature.shape[0]) #[list of [seq_len,nfeats]]
        sample_list = []
        for seq in splitted[0]:
            all_features = seq
            Datastruct = transform.SlimDatastruct
            datastruct = Datastruct(features=all_features)
            sample = datastruct.joints

            sample_list.append(sample.permute(1, 2, 0).unsqueeze(0))
        sample = torch.cat(sample_list)
    else:
        raise NotImplementedError("'feats_to_xyz' not implemented for this dataset")
    return sample

def main():
    """
    Main function for FlowMDM text-to-motion generation.
    
    Orchestrates the complete text-to-motion generation pipeline:
    1. Loads text instructions or dataset samples
    2. Initializes FlowMDM model and diffusion wrapper  
    3. Performs diffusion sampling with BPE and classifier-free guidance
    4. Converts generated features to 3D motion coordinates
    5. Creates MP4 visualizations and saves numpy results
    
    The function supports two modes:
    - Instructions mode: Uses JSON file with text descriptions and lengths
    - Dataset mode: Samples random prompts from training/test data
    
    Parameters
    ----------
    None
        Arguments are parsed from command line using generate_args().
        
    Key Command Line Arguments:
    - model_path: Path to pretrained FlowMDM model (.pt file)
    - instructions_file: JSON with {"text": [...], "lengths": [...]} 
    - bpe_denoising_step: BPE transition point (30-1000, affects quality/smoothness)
    - guidance_param: Text conditioning strength (0.0-5.0, typically 1.5-2.5)
    - num_repetitions: Number of generation samples per prompt
    - use_chunked_att: Memory optimization for long sequences
    
    Returns
    -------
    None
        Outputs are saved to disk:
        - results.npy: Motion data, text prompts, lengths
        - sample_rep*.mp4: Individual motion visualizations  
        - sample_all.mp4: Combined visualization grid
        - prompted_texts.json: Input text descriptions used
    
    Raises
    ------
    AssertionError
        If instructions file format is invalid or model/dataset loading fails.
    FileNotFoundError
        If model file or instructions file doesn't exist.
    RuntimeError
        If CUDA memory issues or model initialization fails.
        
    Notes
    -----
    Motion Generation Workflow:
    1. Text Encoding: CLIP processes text descriptions into embeddings
    2. Diffusion Sampling: BPE-scheduled denoising with text guidance
    3. Feature Conversion: Model features → 3D joint coordinates
    4. Visualization: 3D coordinates → MP4 animations with skeleton
    
    BPE (Blended Positional Encodings) Schedule:
    - Early steps: Absolute PE for global coherence
    - Later steps: Relative PE for smooth transitions  
    - Transition point controlled by bpe_denoising_step parameter
    
    Classifier-Free Guidance:
    - Balances conditioned vs unconditioned generation
    - Higher guidance_param = stronger text adherence
    - guidance_param=0: Unconditional, guidance_param=1: Standard diffusion
    
    Examples
    --------
    # Generate from instruction file (recommended)
    python -m runners.generate \
        --model_path ./results/humanml/FlowMDM/model000500000.pt \
        --instructions_file ./runners/jsons/composition_humanml.json \
        --bpe_denoising_step 60 --guidance_param 2.5 --num_repetitions 3
        
    # Generate from dataset samples
    python -m runners.generate \
        --model_path ./results/babel/FlowMDM/model001300000.pt \
        --num_samples 5 --bpe_denoising_step 125 --guidance_param 1.5
    """
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    fps = datasets_fps[args.dataset]
    assert args.instructions_file == '' or 'json' == args.instructions_file.split('.')[-1], "Instructions file must be a json file"
    dist_util.setup_dist(args.device)
    if out_path == '': # if unspecified, save in the same folder as the model
        out_path = os.path.join(os.path.dirname(args.model_path),
                                '{}_s{}'.format(niter, args.seed))
        if args.instructions_file != '':
            out_path += '_' + os.path.basename(args.instructions_file).replace('.json', '').replace(' ', '_').replace('.', '')

    animation_out_path = out_path
    os.makedirs(animation_out_path, exist_ok=True)

    # ================= Load texts + lengths and adapt batch size ================
    # FlowMDM supports two input modes:
    # 1. Instructions mode: JSON file with text descriptions and frame lengths
    # 2. Dataset mode: Random sampling from training/validation dataset  
    # This block must be called BEFORE the dataset is loaded to set correct batch size
    is_using_data = args.instructions_file == ''
    if not is_using_data: 
        assert os.path.exists(args.instructions_file)
        # load json
        with open(args.instructions_file, 'r') as f:
            instructions = json.load(f)
            assert "text" in instructions and "lengths" in instructions, "Instructions file must contain 'text' and 'lengths' keys"
            assert len(instructions["text"]) == len(instructions["lengths"]), "Instructions file must contain the same number of 'text' and 'lengths' elements"
        num_instructions = len(instructions["text"])
        args.batch_size = num_instructions
        args.num_samples = 1
    else:
        num_instructions = args.num_samples
        args.batch_size = num_instructions
        args.num_samples = 1

    # ================= Load dataset or prepare model_kwargs for inference ================
    # model_kwargs contains all conditioning information for the diffusion process:
    # - y['text']: Text descriptions for each motion segment  
    # - y['lengths']: Frame count for each motion segment
    # - y['mask']: Attention mask for variable-length sequences
    # - y['scale']: Classifier-free guidance scaling factor
    if is_using_data:
        print('Loading dataset...')
        if args.split == "test" and args.dataset == "babel":
            args.split = "val" # Babel does not have a test set

        try:
            data = load_dataset(args, args.split)
        except Exception as e:
            print(f'Error while loading dataset: {e}')
            return
        
        if is_using_data:
            iterator = iter(data)
            sample_gt, model_kwargs = next(iterator)
            
        j = { "sequence": [] }
        for i in range(num_instructions):
            length = model_kwargs['y']['lengths'][i].item()
            text = model_kwargs['y']['text'][i]
            j["sequence"].append([length, text])
        with open(os.path.join(animation_out_path, "prompted_texts.json"), "w") as f:
            json.dump(j, f)

        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
    else: # from instructions
        json_lengths = instructions["lengths"]
        json_texts = instructions["text"]
        mask = torch.ones((len(json_texts), max(json_lengths)))
        for i, length in enumerate(json_lengths):
            mask[i, length:] = 0
        model_kwargs = {'y': {
            'mask': mask,
            'lengths': torch.tensor(json_lengths),
            'text': list(json_texts),
            'tokens': [''],
        }}
        with open(os.path.join(animation_out_path, "prompted_texts.json"), "w") as f:
            json.dump(instructions, f)
            
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

    print(list(zip(list(model_kwargs['y']['text']), list(model_kwargs['y']['lengths'].cpu().numpy()))))

    # ================= Load model and diffusion wrapper ================  
    # Load pretrained FlowMDM model and wrap with diffusion sampling logic
    # DiffusionWrapper adds BPE scheduling and classifier-free guidance
    print("Creating model and diffusion...")
    model, diffusion = load_model(args, dist_util.dev())
    diffusion = DiffusionWrapper(args, diffusion, model)

    # ================= Diffusion Sampling ================
    # Generate multiple repetitions of motion sequences using iterative denoising
    # Each repetition uses the same text prompts but different noise initialization
    all_motions = []
    all_lengths = []
    all_text = []
    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetition #{rep_i}]')
        # Perform iterative denoising to generate motion features
        # BPE schedule controls APE→RPE transition during sampling process
        # Classifier-free guidance balances text conditioning vs diversity  
        sample = diffusion.p_sample_loop(
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
        )
        # Convert model features to 3D joint coordinates for visualization
        sample = feats_to_xyz(sample, args.dataset)

        c_text = ""
        for i in range(num_instructions):
            c_text += model_kwargs['y']['text'][i] + " /// "

        all_text.append(c_text)
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].sum().unsqueeze(0))#.cpu().numpy())

        print(f"created {rep_i+1}/{args.num_repetitions} human motion compositions.")

    all_motions = np.concatenate(all_motions, axis=0)
    all_lengths = np.concatenate(all_lengths, axis=0)

    # ================= Save results + visualizations ================
    # Save generated motion data as numpy arrays and create MP4 visualizations
    # Results include: motion coordinates, text prompts, sequence lengths, metadata
    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
            'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    # Create 3D skeletal animations using T2M kinematic chain (22 joints)
    # Each motion segment gets its own text overlay and timing
    skeleton = paramUtil.t2m_kinematic_chain
    
    sample_print_template, row_print_template, \
    sample_file_template, row_file_template = construct_template_variables(args.unconstrained)

    try:
        rep_files = []
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.num_samples]
            motion = all_motions[rep_i*args.num_samples].transpose(2, 0, 1)
            save_file = sample_file_template.format(rep_i)
            print(sample_print_template.format(rep_i, save_file))
            animation_save_path = os.path.join(animation_out_path, save_file)
            lengths_list = model_kwargs['y']['lengths']
            captions_list = []
            for c, l in zip(caption.split(" /// "), lengths_list):
                captions_list += [c,] * l
            plot_3d_motion_mix(animation_save_path, skeleton, motion, dataset=args.dataset, title=captions_list, fps=fps,
                        vis_mode='alternate', lengths=lengths_list)
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)
    except Exception as e:
        print(f'Error while processing sample: {e}')

    save_multiple_samples(args, animation_out_path,
                                            row_print_template, row_file_template,
                                            caption, rep_files)

    abs_path = os.path.abspath(animation_out_path)
    print(f'[Done] Results are at [{abs_path}]')


def save_multiple_samples(args, out_path, row_print_template, row_file_template, caption, rep_files):
    """
    Combine multiple motion video samples into a single grid visualization.
    
    Uses FFmpeg to horizontally stack individual motion videos (sample_rep*.mp4) 
    into a single combined video showing all repetitions side by side.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing num_repetitions and other settings.
    out_path : str
        Output directory path where the combined video will be saved.
    row_print_template : str  
        Format string for printing the output filename.
    row_file_template : str
        Template for the combined video filename (typically 'sample_all.mp4').
    caption : str
        Text caption describing the motion (currently unused).
    rep_files : list of str
        List of individual video file paths to combine.

    Returns
    -------
    None
        Creates a combined MP4 video file in the output directory.

    Notes
    -----
    - Requires FFmpeg to be installed and available in system PATH
    - Uses horizontal stacking (hstack) filter for side-by-side layout
    - Single repetitions are saved without stacking
    - Failed FFmpeg calls may not raise exceptions but print to stderr
    
    Examples
    --------
    >>> rep_files = ['sample_rep00.mp4', 'sample_rep01.mp4', 'sample_rep02.mp4']
    >>> save_multiple_samples(args, './output/', '[all reps | -> {}]', 
    ...                       'sample_all.mp4', caption, rep_files)
    # Creates: ./output/sample_all.mp4 with 3 videos side by side
    """
    all_rep_save_file = row_file_template
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(all_rep_save_file))


def construct_template_variables(unconstrained):
    """
    Create filename and print templates for motion video output files.
    
    Generates consistent naming conventions for individual sample videos
    and combined multi-repetition videos based on generation mode.

    Parameters
    ----------
    unconstrained : bool
        Whether the model was trained unconditionally (without text conditioning).
        Affects the capitalization in print templates for clarity.

    Returns
    -------
    sample_print_template : str
        Format string for printing individual sample progress.
        Format: '[rep #XX | -> filename]' or '[Rep #XX | -> filename]'
    row_print_template : str
        Format string for printing combined video progress.
        Format: '[all repetitions | -> filename]'
    sample_file_template : str
        Filename template for individual samples.
        Format: 'sample_rep{:02d}.mp4' (e.g., sample_rep00.mp4)
    row_file_template : str
        Filename for combined multi-repetition video.
        Fixed: 'sample_all.mp4'

    Notes
    -----
    - All templates use zero-padded 2-digit repetition numbers
    - The unconstrained parameter only affects print message capitalization
    - File naming is consistent regardless of conditioning mode
    
    Examples
    --------
    >>> templates = construct_template_variables(unconstrained=False)
    >>> sample_print, row_print, sample_file, row_file = templates
    >>> sample_file.format(0)  # 'sample_rep00.mp4'
    >>> sample_print.format(0, 'sample_rep00.mp4')  # '[Rep #00 | -> sample_rep00.mp4]'
    """
    row_file_template = 'sample_all.mp4'
    if unconstrained:
        sample_file_template = 'sample_rep{:02d}.mp4'
        sample_print_template = '[rep #{:02d} | -> {}]'
        row_print_template = '[all repetitions | -> {}]'
    else:
        sample_file_template = 'sample_rep{:02d}.mp4'
        sample_print_template = '[Rep #{:02d} | -> {}]'
        row_print_template = '[all repetitions | -> {}]'

    return sample_print_template, row_print_template, \
           sample_file_template, row_file_template


def load_dataset(args, split):
    """
    Load motion dataset for sampling text prompts and motion sequences.
    
    Creates a PyTorch DataLoader for either HumanML3D or Babel datasets
    configured for generation mode (as opposed to evaluation mode).

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing dataset configuration:
        - dataset: 'humanml' or 'babel'  
        - batch_size: Number of samples per batch
        - min_seq_len, max_seq_len: Sequence length bounds (Babel only)
        - protocol: Dataset protocol/version
        - pose_rep: Pose representation format
    split : str
        Dataset split to load ('train', 'val', 'test').
        Note: Babel uses 'val' instead of 'test'.

    Returns
    -------
    torch.utils.data.DataLoader
        Configured DataLoader with motion samples and text descriptions.
        Each batch contains:
        - Motion features: [batch_size, n_feats, 1, seq_len]
        - Text descriptions: List of strings
        - Sequence lengths: [batch_size] tensor
        - Attention masks: [batch_size, max_seq_len] tensor

    Notes
    -----
    Dataset Configurations:
    - HumanML3D: Fixed 150 frames, 263-dim features (rotation-invariant coordinates)
    - Babel: Variable length (min_seq_len to max_seq_len), 135-dim SMPL parameters
    
    The DataLoader is configured for generation mode which:
    - Loads text descriptions alongside motion data
    - Uses single worker to avoid multiprocessing issues
    - Applies dataset-specific normalization and preprocessing
    
    Examples
    --------
    >>> args.dataset = 'humanml'
    >>> args.batch_size = 4
    >>> data = load_dataset(args, 'test')
    >>> sample_gt, model_kwargs = next(iter(data))
    >>> model_kwargs['y']['text']  # ['person walks forward', ...]
    >>> model_kwargs['y']['lengths']  # tensor([150, 150, 150, 150])
    """
    n_frames = 150 # this comes from PriorMDM, so I'm using it here as well
    if args.dataset == 'babel':
        args.num_frames = (args.min_seq_len, args.max_seq_len)
    else:
        args.num_frames = n_frames

    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=args.num_frames,
                              split=split,#split,
                              load_mode='gen',#'eval',
                              protocol=args.protocol,
                              pose_rep=args.pose_rep,
                              num_workers=1)
    data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
