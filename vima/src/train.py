import os
import torch
from data_loader import VimaDataset
from torch.utils.data import Dataset, DataLoader

from train_utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "true"

_kwargs = {
    "single_word": True,
    "lstrip": False,
    "rstrip": False,
    "normalized": True,
}


def train(cfg):
    seed = 42
    policy = create_policy_from_ckpt(cfg.ckpt, cfg.device).to(cfg.device)
    for epoch in range(cfg.epochs):
        dataset = VimaDataset(root_dir=cfg.data_dir)
        train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
        # 初始化一个optimizer
        optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
        # 初始化一个scheduler，使用warmup
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: min(1.0, x / cfg.warmup_steps))

        for batch in train_loader:
            # batch = batch.to(cfg.device)

            predicted_action = train_one_step(batch, cfg, policy)
            loss = mse_loss(batch['target_action'], predicted_action)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
            if epoch % 100 == 0:
                torch.save(policy.state_dict(), os.path.join(cfg.save_dir, "policy_{}.ckpt".format(epoch)))
            scheduler.step()

# write a mse loss between target_action and predicted_action
def mse_loss(target_action, predicted_action):
    return torch.mean((target_action - predicted_action) ** 2)

def train_one_step(batch, cfg, policy):
    inference_cache = {}
    prompt = batch["prompt"]
    prompt_assets = batch["prompt_assets"]
    obs = batch["obs_img"]
    meta_info = batch["meta_info"]
    obs['ee'] = batch["end_effector"]
    obs["ee"] = np.asarray(obs["ee"])
    elapsed_steps = 0
    # obs = add_batch_dim(obs)
    prompt_token_type, word_batch, image_batch = prepare_prompt(
        prompt=prompt, prompt_assets=prompt_assets, views=["front", "top"]
    )
    word_batch = word_batch.to(cfg.device)
    image_batch = image_batch.to_torch_tensor(device=cfg.device)
    prompt_tokens, prompt_masks = policy.forward_prompt_assembly(
        (prompt_token_type, word_batch, image_batch)
    )
    inference_cache["obs_tokens"] = []
    inference_cache["obs_masks"] = []
    inference_cache["action_tokens"] = []

    obs = prepare_obs(obs=obs, rgb_dict=None, meta=meta_info).to_torch_tensor(
        device=cfg.device
    )
    obs_token_this_step, obs_mask_this_step = policy.forward_obs_token(obs)
    obs_token_this_step = obs_token_this_step.squeeze(0)
    obs_mask_this_step = obs_mask_this_step.squeeze(0)
    inference_cache["obs_tokens"].append(obs_token_this_step[0])
    inference_cache["obs_masks"].append(obs_mask_this_step[0])
    max_objs = max(x.shape[0] for x in inference_cache["obs_tokens"])
    obs_tokens_to_forward, obs_masks_to_forward = [], []
    obs_tokens_this_env, obs_masks_this_env = [], []
    for idx in range(len(inference_cache["obs_tokens"])):
        obs_this_env_this_step = inference_cache["obs_tokens"][idx]
        obs_mask_this_env_this_step = inference_cache["obs_masks"][idx]
        required_pad = max_objs - obs_this_env_this_step.shape[0]
        obs_tokens_this_env.append(
            any_concat(
                [
                    obs_this_env_this_step,
                    torch.zeros(
                        required_pad,
                        obs_this_env_this_step.shape[1],
                        device=cfg.device,
                        dtype=obs_this_env_this_step.dtype,
                    ),
                ],
                dim=0,
            )
        )
        obs_masks_this_env.append(
            any_concat(
                [
                    obs_mask_this_env_this_step,
                    torch.zeros(
                        required_pad,
                        device=cfg.device,
                        dtype=obs_mask_this_env_this_step.dtype,
                    ),
                ],
                dim=0,
            )
        )
    obs_tokens_to_forward.append(any_stack(obs_tokens_this_env, dim=0))
    obs_masks_to_forward.append(any_stack(obs_masks_this_env, dim=0))
    obs_tokens_to_forward = any_stack(obs_tokens_to_forward, dim=0)
    obs_masks_to_forward = any_stack(obs_masks_to_forward, dim=0)
    obs_tokens_to_forward = obs_tokens_to_forward.transpose(0, 1)
    obs_masks_to_forward = obs_masks_to_forward.transpose(0, 1)

    if elapsed_steps == 0:
        action_tokens_to_forward = None
    else:
        action_tokens_to_forward = any_stack(
            [any_stack(inference_cache["action_tokens"], dim=0)],
            dim=0,
        )
        action_tokens_to_forward = action_tokens_to_forward.transpose(0, 1)
    predicted_action_tokens = policy.forward(
        obs_token=obs_tokens_to_forward,
        action_token=action_tokens_to_forward,
        prompt_token=prompt_tokens,
        prompt_token_mask=prompt_masks,
        obs_mask=obs_masks_to_forward,
    )  # (L, B, E)
    predicted_action_tokens = predicted_action_tokens[-1].unsqueeze(
        0
    )  # (1, B, E)

    dist_dict = policy.forward_action_decoder(predicted_action_tokens)
    actions = {k: v.mode() for k, v in dist_dict.items()}
    action_tokens = policy.forward_action_token(actions)  # (1, B, E)
    action_tokens = action_tokens.squeeze(0)  # (B, E)
    inference_cache["action_tokens"].append(action_tokens[0])
    actions = policy._de_discretize_actions(actions)
    action_bounds = [meta_info["action_bounds"]]
    action_bounds_low = [action_bound["low"] for action_bound in action_bounds]
    action_bounds_high = [
        action_bound["high"] for action_bound in action_bounds
    ]
    action_bounds_low = np.asarray(action_bounds_low)
    action_bounds_high = np.asarray(action_bounds_high)
    action_bounds_low = torch.tensor(
        action_bounds_low, dtype=torch.float32, device=cfg.device
    )
    action_bounds_high = torch.tensor(
        action_bounds_high, dtype=torch.float32, device=cfg.device
    )
    actions["pose0_position"] = (
            actions["pose0_position"] * (action_bounds_high - action_bounds_low)
            + action_bounds_low
    )
    actions["pose1_position"] = (
            actions["pose1_position"] * (action_bounds_high - action_bounds_low)
            + action_bounds_low
    )
    actions["pose0_position"] = torch.clamp(
        actions["pose0_position"], min=action_bounds_low, max=action_bounds_high
    )
    actions["pose1_position"] = torch.clamp(
        actions["pose1_position"], min=action_bounds_low, max=action_bounds_high
    )
    actions["pose0_rotation"] = actions["pose0_rotation"] * 2 - 1
    actions["pose1_rotation"] = actions["pose1_rotation"] * 2 - 1
    actions["pose0_rotation"] = torch.clamp(
        actions["pose0_rotation"], min=-1, max=1
    )
    actions["pose1_rotation"] = torch.clamp(
        actions["pose1_rotation"], min=-1, max=1
    )
    actions = {k: v.cpu().numpy() for k, v in actions.items()}
    actions = any_slice(actions, np.s_[0, 0])

    return actions




def parse_args():
    arg = argparse.ArgumentParser()
    arg.add_argument("--data_dir", type=str, default="/Users/lesjie/PycharmProjects/vima/vimadata/vima_v6")
    arg.add_argument("--task", type=str, default="rearrange_then_restore")
    arg.add_argument("--pretrain", type=bool, default=False)
    arg.add_argument("--ckpt", type=str, default='../checkpoints/2M.ckpt')
    arg.add_argument("--lr", type=float, default=0.0001)
    arg.add_argument("--warmup_steps", type=int, default=7000)
    arg.add_argument("--epochs", type=int, default=100)
    arg.add_argument("--device", default="cpu")
    arg = arg.parse_args()
    return arg

if __name__ == '__main__':
    args = parse_args()
    train(args)


