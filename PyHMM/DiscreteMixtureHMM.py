import numpy as np
from scipy.stats import dirichlet


class DiscreteMixtureHMM:
    """
    Base class for HMMs with emissions modelled as discrete mixtures
    Based on GMM-HMMs, for eg here: https://www.inf.ed.ac.uk/teaching/courses/asr/2012-13/asr03-hmmgmm-4up.pdf
    """

    def __init__(self, num_states=None, num_mixture_components=None,
                 max_iter=1000, threshold=1e-4, num_emissions=None):
        """
        Initialize the model with the basic parameters
        :param num_states: Number of states
        :param num_mixture_components: Number of emission mixture components in each state (assumed to be equal, int)
        :param max_iter: Maximum number of iterations for EM. Default = 1000
        :param threshold: Convergence threshold for log likelihood/log posterior. Default = 1e-4
        :param num_emissions: Number of emissions
        """
        self.num_states = int(num_states)
        self.num_mixture_components = num_mixture_components
        self.num_emissions = int(num_emissions)
        self.max_iter = int(max_iter)
        self.threshold = threshold
        self.log_posterior = []

    def forward(self):
        """
        Run the forward message passing algorithm for the E step
        :return: alpha: Normalized forward messages of shape (num_states, num_mixture_components, shape of data)
        :return: scaling: Normalizing constant of forward messages (also called scaling factors) - shape same as data
        """
        alpha = np.zeros(self.state_and_component_likelihood.shape)
        scaling = np.zeros(self.state_and_component_likelihood.shape[2:])
        for sequence in range(alpha.shape[2]):
            alpha[:, :, sequence, 0] = self.p_start[:, None] * self.p_mixing * \
                                       self.state_and_component_likelihood[:, :, sequence, 0]
            scaling[sequence, 0] = np.sum(alpha[:, :, sequence, 0])
            alpha[:, :, sequence, 0] /= scaling[sequence, 0]
            for time in range(1, alpha.shape[3], 1):
                # For alpha(t-1), sum out the mixture components to get the total forward message for each state
                alpha_summed_previous = np.sum(alpha[:, :, sequence, time - 1], axis=1)
                alpha[:, :, sequence, time] = np.sum(alpha_summed_previous[:, None] *
                                                     self.p_transitions, axis=0)[:, None] * \
                                              self.p_mixing * self.state_and_component_likelihood[:, :, sequence, time]
                scaling[sequence, time] = np.sum(alpha[:, :, sequence, time])
                alpha[:, :, sequence, time] /= scaling[sequence, time]

        return alpha, scaling

    def backward(self, scaling):
        """
        Run the backward message passing algorithm for the E step
        :param scaling: Scaling factors from the forward algorithm
        :return: beta: Normalized backward messages of shape (num_states, num_mixture_components, shape of data)
        """
        beta = np.zeros(self.state_and_component_likelihood.shape)
        for sequence in range(beta.shape[2]):
            beta[:, :, sequence, -1] = np.ones(self.state_and_component_likelihood.shape[:2])
            for time in range(beta.shape[3] - 2, -1, -1):
                # For beta(t+1), sum out the mixture components to get the total backward message for each state
                beta[:, :, sequence, time] = np.sum(np.sum(beta[:, :, sequence, time + 1] * self.p_mixing *
                                                    self.state_and_component_likelihood[:, :, sequence, time + 1], axis=1)[None, :] *
                                                    self.p_transitions, axis=1)[:, None]
                beta[:, :, sequence, time] /= scaling[sequence, time + 1]

        return beta

    def E_step(self):
        """
        Run the E step of the EM algorithm, consisting of the forward and backward message passing algorithms
        :return: alpha: Normalized forward messages of shape (num_states, num_mixture_components, shape of data)
        :return: beta: Normalized backward messages of shape (num_states, num_mixture_components, shape of data)
        :return: scaling: Normalizing constant of forward messages (also called scaling factors)
        :return: expected_latent_state_and_component: Posterior probability of a single state
                                                      and emission mixture component at a time point.
                 Shape (num_states, num_mixture_components, shape of data). Should sum to 1.0 along (0th+1st) axis
        :return: expected_latent_state_pair: Posterior probability of pairs of states.
                 Shape (num_states, num_states, shape of data). Should sum to 1.0 along 0th and 1st axes.

        """
        alpha, scaling = self.forward()
        beta = self.backward(scaling)

        expected_latent_state_and_component = alpha * beta
        # The forward (marginal) message of states at each time point (summing over mixture components)
        states_forward = np.sum(alpha, axis=1)
        # The backward (marginal) message of states at each time point (summing over mixture components)
        states_backward = np.sum(beta * self.p_mixing[:, :, None, None] * self.state_and_component_likelihood, axis=1)
        # Multiply these marginal messages with the transition probability to get expectation of latent state pairs
        expected_latent_state_pair = states_forward[:, None, :, :-1] * \
                                     self.p_transitions[:, :, None, None] * \
                                     states_backward[None, :, :, 1:] / scaling[None, None, :, 1:]

        # Assert that expected_latent_state and expected_latent_state_pair sum up to 1.0 appropriately
        np.testing.assert_allclose(np.sum(expected_latent_state_and_component, axis=(0, 1)),
                                   np.ones(expected_latent_state_and_component.shape[2:]))
        np.testing.assert_allclose(np.sum(expected_latent_state_pair, axis=(0, 1)),
                                   np.ones(expected_latent_state_pair.shape[2:]))

        return alpha, beta, scaling, expected_latent_state_and_component, expected_latent_state_pair


class BernoulliMixtureHMM(DiscreteMixtureHMM):
    """
    Class for HMM with Bernoulli mixture emissions.
    """

    def __init__(self, num_states=None, num_mixture_components=None,
                 max_iter=1000, threshold=1e-4, num_emissions=None):
        """
        Initialize the categorical HMM using the __init__ method of DiscreteHMM
        :param num_states: Number of states
        :param max_iter: Maximum number of iterations for EM. Default = 1000
        :param threshold: Convergence threshold for log likelihood/log posterior. Default = 1e-4
        :param num_emissions: Number of emissions
        """
        super(BernoulliMixtureHMM, self).__init__(num_states, num_mixture_components,
                                                  max_iter, threshold, num_emissions)

    def get_state_and_component_likelihood(self, data):
        """
        Get the likelihood of the data in each latent state (and for each emission mixture component)
        under the current set of parameters
        :param data: Training data
        :return:
        """
        self.state_and_component_likelihood = (self.p_emissions[:, :, :, None, None] ** data[None, None, :, :, :]) \
                            * ((1.0 - self.p_emissions[:, :, :, None, None]) ** (1.0 - data[None, None, :, :, :]))
        self.state_and_component_likelihood = np.prod(self.state_and_component_likelihood, axis=2)

    def M_step(self, data, expected_latent_state_and_component, expected_latent_state_pair):
        """
        Get the parameter settings that maximize the expected log likelihood/log posterior coming from the E step.
        :param data: Training data. Shape = (num_emissions X number of sequences X length of sequence)
        :param expected_latent_state_and_component: Posterior probability of a single mixture component of a single
                 state at a time point. Shape (num_states, num_mixture_components, shape of data[1:]).
        :param expected_latent_state_pair: Posterior probability of pairs of states.
                 Shape (n_states, n_states, shape of data[1:]).
        :return:
        """
        start_count = np.sum(expected_latent_state_and_component[:, :, :, 0], axis=(1, 2))
        start_count += self.start_pseudocounts
        self.p_start = start_count / np.sum(start_count)

        transition_count = np.sum(expected_latent_state_pair, axis=(2, 3))
        transition_count += self.transition_pseudocounts
        self.p_transitions = transition_count / np.sum(transition_count, axis=1, keepdims=True)

        component_count = np.sum(expected_latent_state_and_component, axis=(2, 3))
        component_count += self.component_pseudocounts
        self.p_mixing = component_count / np.sum(component_count, axis=1, keepdims=True)

        emission_count = np.sum(expected_latent_state_and_component[:, :, None, :, :] *
                                data[None, None, :, :, :], axis=(3, 4))
        emission_count += self.emission_pseudocounts[:, :, :, 0]
        total_count = np.sum(expected_latent_state_and_component, axis=(2, 3))[:, :, None] + \
                      np.sum(self.emission_pseudocounts, axis=3)
        self.p_emissions = emission_count / total_count

    def fit(self, data, p_mixing, p_transitions, p_emissions, p_start, start_pseudocounts, transition_pseudocounts,
            component_pseudocounts, emission_pseudocounts, verbose=True):
        """
        Run the EM algorithm to find the maximum likelihood or maximum a posteriori (if pseudocounts >0) estimates
        of the model parameters
        :param data: Training data. Shape = (num_emissions X number of sequences X length of sequence)
        :param p_mixing: Initial guess for emission component mixing matrix.
                         Shape = (num_states X num_mixture_components)
        :param p_transitions: Initial guess for transition probability matrix. Shape = (num_states X num_states)
        :param p_emissions: Initial guess for emission probability matrix.
                            Shape = (num_states X num_mixture_components X num_emissions)
        :param p_start: Initial guess for first state occupancy probability matrix. Shape = (num_states)
        :param start_pseudocounts: Parameters for Dirichlet prior on first state occupancy. Shape = (num_states)
        :param transition_pseudocounts: Parameters for Dirichlet priors on transition probabilities.
               Shape = (num_states X num_states)
        :param component_pseudocounts: Parameters for Dirichlet priors on emission distribution mixing probabilities
                                       Shape = (num_states X num_mixture_components)
        :param emission_pseudocounts: Parameters for Beta priors on emission probabilities. The emissions consist of
               (num_mixture_components X num_emission) Bernoulli mixture distributed emissions.
               Hence their Beta priors have 2 parameters each
               Shape = (num_states X num_mixture_components X num_emissions X 2)
        :param verbose: Show the improvement in log likelihood/log posterior through training. Default = True
        :return:
        """
        self.p_mixing = p_mixing / np.sum(p_mixing, axis=1, keepdims=True)
        self.p_transitions = p_transitions / np.sum(p_transitions, axis=1, keepdims=True)
        self.p_emissions = p_emissions
        self.p_start = p_start / np.sum(p_start)
        self.start_pseudocounts = start_pseudocounts
        self.transition_pseudocounts = transition_pseudocounts
        self.component_pseudocounts = component_pseudocounts
        self.emission_pseudocounts = emission_pseudocounts
        self.converged = False

        for iter in range(self.max_iter):
            self.get_state_and_component_likelihood(data.astype("int"))
            alpha, beta, scaling, expected_latent_state_and_component, expected_latent_state_pair = self.E_step()
            self.M_step(data.astype("int"), expected_latent_state_and_component, expected_latent_state_pair)
            current_log_likelihood = np.sum(np.log(scaling))
            current_log_prior = 0
            for state in range(self.num_states):
                current_log_prior += dirichlet.logpdf(self.p_transitions[state, :],
                                                      self.transition_pseudocounts[state, :])
                current_log_prior += dirichlet.logpdf(self.p_mixing[state, :],
                                                      self.component_pseudocounts[state, :])
                for component in range(self.num_mixture_components):
                    for emission in range(self.num_emissions):
                        current_log_prior += dirichlet.logpdf([self.p_emissions[state, component, emission],
                                                               1 - self.p_emissions[state, component, emission]],
                                                              self.emission_pseudocounts[state, component, emission, :])
            current_log_prior += dirichlet.logpdf(self.p_start, self.start_pseudocounts)
            self.log_posterior.append(current_log_likelihood + current_log_prior)

            if iter >= 1:
                improvement = self.log_posterior[-1] - self.log_posterior[-2]
                if verbose:
                    print("Training improvement in log posterior:", improvement)
                if improvement <= self.threshold:
                    self.converged = True
                    break
