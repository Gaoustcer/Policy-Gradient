implementation of Actor-Critic Methods
# The Actor-Critic Methods
## Net
Actor Net $\pi_\theta(s)$ output the distribution of Action over action space.
Critic Net $Q(s,a,\omega)$ this is used for validation of the value of current state and action
Critic Target Net $Q(s,a,\hat\omega)$ for calculate the TD error of a transition
## Algorithm Iteration
1. Sample the action from Actor Net $a\in \pi(a|s)$
2. use Critic Net to update parameter of Actor
3. Get the act result for the action you choose and get a Transition $(s,a,r,s^\prime)$
4. Calculate the TD error to update the parameter of Critic Net
5. update the Target Critic Net

## Question
Shared Feature Extraction of Actor and Critic or Not?
Policy Gradient we use must be on-policy which indicates all experience used to update the policy is collected by the policy itself