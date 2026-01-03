# HIR: Hindsigh Instruction Relabeling
- https://github.com/tianjunz/HIR
- https://arxiv.org/pdf/2302.05206.pdf

After I read the code here is my understanding. It has two modes:
- 1) online exploration (here we generate new trajectories) which leads to the final goal.
  For example answer of the question
- 2.1) After collecting a few unique trajectories check if they are valid (is the conclusion correct)
- 2.2) Normal finetune the model on the examples + If the reasoning path is wrong, then treat it as an additional loss
This acts like PPO, only we don't need the reward model.

## Two-stage Reinforcement Learning

Định nghĩa Markov Decision Process (MDP) <S,A,P,R> với:
- S là không gian trạng thái
- A là không gian hành động
- P là xác xuất chuyển trạng thái P(s'|s, a)
- R(s, a) là hàm phần thưởng

Mục đích của RL là tìm một chính sách tối ưu pi^star làm tối ưu hóa kỳ vọng của phần thưởng tích lũy:
`J(pi) = E_pi [ sum_{t=0}^infinity( y^t R(s_t, a_t) ) ]` với a_t ~ pi(a|s_t)