import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

from torch.utils.tensorboard.writer import SummaryWriter

from tqdm import tqdm

from dqn.BaseDQN import *
REND_VEDIO_EPISODE = [5, 50, 100, 200, 300, 400, 600, 790]

def train(name: str, episode: int = 500, hparam: HyperParam | None = None, is_write: bool = True):
    env = gym.make(
        "CartPole-v0", 
        render_mode = "rgb_array"
    )
    env = RecordEpisodeStatistics(env, buffer_length = 1)
    if is_write:
        env = RecordVideo(
            env, 
            video_folder = "vedio_CartPole_with_BaseDQN", 
            name_prefix = name,
            episode_trigger = lambda x: (x + 1) % 100 == 0
        )

    if hparam == None:
        hparam = HyperParam()
    model = BaseDQN(hparam)

    writer = None
    if is_write:
        writer = SummaryWriter(comment = name)

    for episode in tqdm(range(episode)):
        state, info = env.reset()
        done = False
        total_loss = 0

        while not done:
            
            action = model.take_action_single(state)
            next_state, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated

            transition = make_transition_from_numpy(state, action, next_state, reward, done)
            loss = model.update(transition)
            if loss != False:
                total_loss += loss

            state = next_state
        
        if writer != None:        
            writer.add_scalar(
                f"{name}/avg_loss",
                total_loss / info["episode"]["l"],
                episode
            )
            writer.add_scalar(
                f"{name}/return",
                info["episode"]["r"],
                episode
            )

        model.update_episode(episode)

        if writer != None:  
            if episode % 20 == 0:
                action_sum = 0
                for i in model.reply_queue.buffer:
                    action_sum += i.action.item()

                writer.add_scalar(
                    f"{name}/avg_action",
                    action_sum / model.reply_queue.size(),
                    int(episode / 20)
                )
    env.close()
    
    if writer != None:
        writer.close()

if __name__ == "__main__":
    # hparam = HyperParam(
    #     reply_size = 100000,
    #     minimal_size = 5000,
    #     alpha = 5e-4,
    #     episode_decay = 200,
    #     epsilon_final = 0,
    #     epsilon_start = 1.0
    # )
    # train("test_run_v1_huge_rs", 1500, is_write = True)

    hparam = HyperParam(
        epsilon_final = 0.01,
        epsilon_start = 1.0
    )
    train("test_run_v0_small_final", 500, is_write = True)

# 记录
# 如果 loss 不断发散, 可能是出现严重的高估问题, 可通过检查经验队列中特定动作是否高频次出现
# 当环境结束时, td target 不需要再需要模型预测 (未来奖励一定是 0)
# epsilon_final 应设为 0, 否则模型即使收敛, 也将极不稳定 
# 虽然模型短时间接近收敛但性能又下降, 可能是出现了未学习过的场景, 再训练一段时间可能收敛
# 完成一次片段的时间越长, 经验队列容量应该越大 ? (大概错误)
# 下一步: 询问, 减小 episode_decay ?, 防止绕动导致的发散 ?
