import numpy as np
from scipy.special import digamma, gammaln
from scipy.optimize import minimize


def KL_Dirichlet(a, b):
    """
    Helper function
    Calculate KL divergence between Dirichlet distributions with parameters in a and b.
    :param a: Parameters of 1st set of Dirichlet distributions. Several distributions can be stacked on 0th axis to give
              shape (num_distributions X num_parameters)
    :param b: Parameters of 2nd set of Dirichlet distributions. Shape same as a
    :return: KL divergence between row-matched pairs of Dirichlet distributions with parameters in a and b.
    """

    # Add an axis to the Dirichlet parameter vectors a and b if they correspond to a single distribution
    # Multiple distributions stacked together will anyways have 2 axes
    if len(a.shape) == 1 and len(b.shape) == 1:
        a = a[None, :]
        b = b[None, :]

    # Taken from http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
    KL = gammaln(np.sum(a, axis = 1, keepdims = True)) - gammaln(np.sum(b, axis = 1, keepdims = True)) \
         - np.sum(gammaln(a), axis = 1, keepdims = True) + np.sum(gammaln(b), axis = 1, keepdims = True) \
         + np.sum((a - b) * (digamma(a) - digamma(np.sum(a, axis = 1, keepdims = True))), axis = 1, keepdims = True)
    return KL


def minimize_KL_Categorical(x, model):
    """
    Helper function to use with scipy.optimize.minimize to minimize KL divergence between posterior and prior Dirichlet
    distributions by optimizing the hyperpriors.
    :param x: 3 element np.array with the hyperpriors. 0th, 1st and 2nd element are the start, transition and emission
    hyperpriors respectively.
    :param model: variationalHMM model
    :return: sum of KL divergences between rows of posterior and prior Dirichlet distributions
    """

    start_hyperprior = x[0]
    transition_hyperprior = x[1]
    emission_hyperprior = x[2]
    KL = np.sum(KL_Dirichlet(model.start_counts, start_hyperprior*np.ones(model.num_states))) \
         + np.sum(KL_Dirichlet(model.transition_counts, transition_hyperprior*np.ones((model.num_states, model.num_states)))) \
         + np.sum(KL_Dirichlet(model.emission_counts, emission_hyperprior*np.ones((model.num_states, model.num_emissions))))
    return KL


def minimize_KL_Bernoulli(x, model):
    """
    Helper function to use with scipy.optimize.minimize to minimize KL divergence between posterior and prior Dirichlet
    distributions by optimizing the hyperpriors.
    :param x: 3 element np.array with the hyperpriors. 0th, 1st and 2nd element are the start, transition and emission
    hyperpriors respectively.
    :param model: variationalHMM model
    :return: sum of KL divergences between rows of posterior and prior Dirichlet distributions
    """

    start_hyperprior = x[0]
    transition_hyperprior = x[1]
    emission_hyperprior = x[2]
    KL = np.sum(KL_Dirichlet(model.start_counts, start_hyperprior * np.ones(model.num_states))) \
         + np.sum(KL_Dirichlet(model.transition_counts, transition_hyperprior
                               * np.ones((model.num_states, model.num_states)))) \
         + np.sum([KL_Dirichlet(model.emission_counts[:, emission, :],
                                emission_hyperprior * np.ones((model.num_states, 2)))
                   for emission in range(model.num_emissions)])
    return KL


class DiscreteHMM:
    """
    Base class for variational inference in HMMs with discrete emissions
    Based on Matthew Beal's thesis (2003), Chapter 3
    https://cse.buffalo.edu/faculty/mbeal/thesis/
    """

    def __init__(self, num_states = None, max_iter = 1000, threshold = 1e-4, num_emissions = None):
        """
        Initialize the model with the basic parameters
        :param num_states: Number of states
        :param max_iter: Maximum number of iterations for EM. Default = 1000
        :param threshold: Convergence threshold for log likelihood/log posterior. Default = 1e-4
        :param num_emissions: Number of emissions
        """
        self.num_states = int(num_states)
        self.num_emissions = int(num_emissions)
        self.max_iter = int(max_iter)
        self.threshold = threshold
        self.ELBO = []

    def forward(self):
        """
        Run the forward message passing algorithm for the E step
        :return: alpha: Normalized forward messages of shape (n_states, shape of data)
        :return: scaling: Normalizing constant of forward messages (also called scaling factors)
        """
        alpha = np.zeros(self.state_likelihood.shape)
        scaling = np.zeros(self.state_likelihood.shape[1:])
        for sequence in range(alpha.shape[1]):
            alpha[:, sequence, 0] = self.p_start * self.state_likelihood[:, sequence, 0]
            scaling[sequence, 0] = np.sum(alpha[:, sequence, 0])
            alpha[:, sequence, 0] /= scaling[sequence, 0]
            for time in range(1, alpha.shape[2], 1):
                alpha[:, sequence, time] = np.sum(alpha[:, sequence, time - 1][:, None] * self.p_transitions, axis = 0) \
                                                * self.state_likelihood[:, sequence, time]
                scaling[sequence, time] = np.sum(alpha[:, sequence, time])
                alpha[:, sequence, time] /= scaling[sequence, time]

        return alpha, scaling

    def backward(self, scaling):
        """
        Run the backward message passing algorithm for the E step
        :param scaling: Scaling factors from the forward algorithm
        :return: beta: Normalized backward messages of shape (n_states, shape of data)
        """
        beta = np.zeros(self.state_likelihood.shape)
        for sequence in range(beta.shape[1]):
            beta[:, sequence, -1] = np.ones(self.num_states)
            for time in range(beta.shape[2] - 2, -1, -1):
                beta[:, sequence, time] = np.sum(np.expand_dims(beta[:, sequence, time + 1]
                                                 * self.state_likelihood[:, sequence, time + 1], axis = 0)
                                                 * self.p_transitions, axis = 1)
                beta[:, sequence, time] /= scaling[sequence, time + 1]

        return beta

    def E_step(self):
        """
        Run the E step of the EM algorithm, consisting of the forward and backward message passing algorithms
        :return: alpha: Normalized forward messages of shape (n_states, shape of data)
        :return: beta: Normalized backward messages of shape (n_states, shape of data)
        :return: scaling: Normalizing constant of forward messages (also called scaling factors)
        :return: expected_latent_state: Posterior probability of a single state at a time point.
                 Shape (n_states, shape of data). Should sum to 1.0 along 0th axis
        :return: expected_latent_state_pair: Posterior probability of pairs of states.
                 Shape (n_states, n_states, shape of data). Should sum to 1.0 along 0th and 1st axes.

        """
        alpha, scaling = self.forward()
        beta = self.backward(scaling)

        expected_latent_state = alpha * beta
        expected_latent_state_pair = alpha[:, None, :, :-1] * self.p_transitions[:, :, None, None] \
                                     * self.state_likelihood[None, :, :, 1:] * beta[None, :, :, 1:] \
                                     / scaling[None, None, :, 1:]

        # Assert that expected_latent_state and expected_latent_state_pair sum up to 1.0 appropriately
        np.testing.assert_allclose(np.sum(expected_latent_state, axis = 0), np.ones(expected_latent_state.shape[1:]))
        np.testing.assert_allclose(np.sum(expected_latent_state_pair, axis = (0, 1)),
                                   np.ones(expected_latent_state_pair.shape[2:]))

        return alpha, beta, scaling, expected_latent_state, expected_latent_state_pair


class CategoricalHMM(DiscreteHMM):
    """
    Class for variational inference in HMMs with categorical emissions that sum to 1.0.
    Based on Matthew Beal's thesis (2003), Chapter 3
    https://cse.buffalo.edu/faculty/mbeal/thesis/
    """

    def __init__(self, num_states = None, max_iter = 1000, threshold = 1e-4, num_emissions = None):
        """
        Initialize the categorical HMM using the __init__ method of DiscreteHMM
        :param num_states: Number of states
        :param max_iter: Maximum number of iterations for EM. Default = 1000
        :param threshold: Convergence threshold for log likelihood/log posterior. Default = 1e-4
        :param num_emissions: Number of emissions
        """
        super(CategoricalHMM, self).__init__(num_states, max_iter, threshold, num_emissions)

    def get_state_likelihood(self, data):
        """
        Get the likelihood of the data in each latent state under the current set of parameters (subnormalized
        probabilities)
        :param data: Training data
        :return:
        """
        self.state_likelihood = self.p_emissions[:, data]

    def get_subnormalized_probabilities(self):
        """
        Calculate subnormalized probabilities, i.e, the parameter distributions averaged over the posterior of hidden
        states. Forward-backward can now be run with these subnormalized probabilities.
        :return:
        """

        # Equations 3.68 to 3.71 in Beal thesis
        self.p_transitions = np.exp(digamma(self.transition_counts)
                                    - digamma(np.sum(self.transition_counts, axis = 1, keepdims = True)))
        self.p_emissions = np.exp(digamma(self.emission_counts)
                                    - digamma(np.sum(self.emission_counts, axis = 1, keepdims = True)))
        self.p_start = np.exp(digamma(self.start_counts) - digamma(np.sum(self.start_counts)))

    def M_step(self, data, expected_latent_state, expected_latent_state_pair):
        """
        Update the start, transition and emission posteriors by adding sufficient statistics of the hidden state
        posterior to the parameter priors.
        :param data: Training data. Shape = (number of sequences X length of sequence)
        :param expected_latent_state: Posterior probability of a single state at a time point.
                 Shape (n_states, shape of data).
        :param expected_latent_state_pair: Posterior probability of pairs of states.
                 Shape (n_states, n_states, shape of data).
        :return:
        """

        # Equations 3.54 to 3.59 in Beal thesis
        self.start_counts = self.prior_start_counts + np.sum(expected_latent_state[:, :, 0], axis = 1)
        self.transition_counts = self.prior_transition_counts + np.sum(expected_latent_state_pair, axis = (2, 3))
        self.emission_counts = np.zeros((self.num_states, self.num_emissions))
        self.emission_counts += self.prior_emission_counts
        for state in range(self.num_states):
            np.add.at(self.emission_counts[state, :], data, expected_latent_state[state, :, :])

    def get_ELBO(self, scaling):
        """
        Calculate the lower bound on marginal likelihood. Only applicable right after E step.
        :param scaling: Scaling factors from the E step.
        :return:
        """

        KL_start = KL_Dirichlet(self.start_counts, self.prior_start_counts)
        KL_transition = KL_Dirichlet(self.transition_counts, self.prior_transition_counts)
        KL_emission = KL_Dirichlet(self.emission_counts, self.prior_emission_counts)

        # Equation 3.79 of Beal thesis
        ELBO = - np.sum(KL_start) - np.sum(KL_transition) - np.sum(KL_emission) + np.sum(np.log(scaling))
        return ELBO

    def update_hyperpriors(self):
        """
        Optimize hyperpriors by minimizing KL divergence between posterior and prior distributions of parameters.
        Uses the minimize_KL_Categorical helper function.
        :return:
        """

        optimizer = minimize(minimize_KL_Categorical, np.array([self.start_hyperprior, self.transition_hyperprior,
                                                    self.emission_hyperprior]), args = (self,),
                             constraints = {"type": "ineq", "fun": lambda x: x})
        self.start_hyperprior = optimizer["x"][0]
        self.transition_hyperprior = optimizer["x"][1]
        self.emission_hyperprior = optimizer["x"][2]

    def fit(self, data, transition_hyperprior, emission_hyperprior, start_hyperprior, initial_transition_counts,
            initial_emission_counts, initial_start_counts, update_hyperpriors = True, update_hyperpriors_iter = 1,
            verbose = True):
        """
        Run the variational inference algorithm in Beal (2003). Assumes symmetric priors on start, transition and
        emission parameters with hyperpriors that are optimized as well.
        :param data: Training data. Shape = (number of sequences X length of sequence)
        :param transition_hyperprior: Hyperprior of transition counts
        :param emission_hyperprior: Hyperprior of emission counts
        :param start_hyperprior: Hyperprior of start counts
        :param initial_transition_counts: Initial guess of posterior transition counts.
               Shape = (num_states X num_states)
        :param initial_emission_counts: Initial guess of posterior emission counts. Shape = (num_states X num_emissions)
        :param initial_start_counts: Initial guess of posterior start counts. Shape = (num_states)
        :param update_hyperpriors: Whether hyperpriors should be updated. Defaults to True.
        :param update_hyperpriors_iter: Number of iterations between hyperprior update steps. Defaults to 1.
        :param verbose: Show the improvement in ELBO through training. Default = True
        :return:
        """

        self.transition_hyperprior = transition_hyperprior
        self.emission_hyperprior = emission_hyperprior
        self.start_hyperprior = start_hyperprior
        self.transition_counts = initial_transition_counts
        self.emission_counts = initial_emission_counts
        self.start_counts = initial_start_counts

        self.converged = False

        for iter in range(self.max_iter):
            self.prior_transition_counts = self.transition_hyperprior * np.ones((self.num_states, self.num_states))
            self.prior_emission_counts = self.emission_hyperprior * np.ones((self.num_states, self.num_emissions))
            self.prior_start_counts = self.start_hyperprior * np.ones(self.num_states)

            self.get_subnormalized_probabilities()
            self.get_state_likelihood(data.astype("int"))
            alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = self.E_step()
            self.ELBO.append(self.get_ELBO(scaling))

            if iter >= 1:
                improvement = self.ELBO[-1] - self.ELBO[-2]
                if verbose:
                    print("Training improvement in ELBO:", improvement)
                if improvement <= self.threshold:
                    self.converged = True
                    break
                if update_hyperpriors:
                    if iter % update_hyperpriors_iter == 0:
                        self.update_hyperpriors()

            self.M_step(data.astype("int"), expected_latent_state, expected_latent_state_pair)


class IndependentBernoulliHMM(DiscreteHMM):
    """
    Class for variational inference for HMMs with independent Bernoulli emissions.
    Based on Matthew Beal's thesis (2003)
    https://cse.buffalo.edu/faculty/mbeal/thesis/
    """

    def __init__(self, num_states = None, max_iter = 1000, threshold = 1e-4, num_emissions = None):
        """
        Initialize the IndependentBernoulli HMM using the __init__ method of DiscreteHMM
        :param num_states: Number of states
        :param max_iter: Maximum number of iterations for EM. Default = 1000
        :param threshold: Convergence threshold for log likelihood/log posterior. Default = 1e-4
        :param num_emissions: Number of emissions
        """
        super(IndependentBernoulliHMM, self).__init__(num_states, max_iter, threshold, num_emissions)

    def get_state_likelihood(self, data):
        """
        Get the likelihood of the data in each latent state under the current set of parameters
        :param data: Training data
        :return:
        """
        self.state_likelihood = (self.p_emissions[:, :, 0][:, :, None, None] ** data[None, :, :, :]) \
                            * (self.p_emissions[:, :, 1][:, :, None, None] ** (1.0 - data[None, :, :, :]))
        self.state_likelihood = np.prod(self.state_likelihood, axis = 1)

    def get_subnormalized_probabilities(self):
        """
        Calculate subnormalized probabilities, i.e, the parameter distributions averaged over the posterior of hidden
        states. Forward-backward can now be run with these subnormalized probabilities.
        :return:
        """

        # Equations 3.68 to 3.71 in Beal thesis
        self.p_transitions = np.exp(digamma(self.transition_counts)
                                    - digamma(np.sum(self.transition_counts, axis = 1, keepdims = True)))
        self.p_emissions = np.exp(digamma(self.emission_counts)
                                    - digamma(np.sum(self.emission_counts, axis = 2, keepdims = True)))
        self.p_start = np.exp(digamma(self.start_counts) - digamma(np.sum(self.start_counts)))

    def get_ELBO(self, scaling):
        """
        Calculate the lower bound on marginal likelihood. Only applicable right after E step.
        :param scaling: Scaling factors from the E step.
        :return:
        """

        KL_start = KL_Dirichlet(self.start_counts, self.prior_start_counts)
        KL_transition = KL_Dirichlet(self.transition_counts, self.prior_transition_counts)
        KL_emission = []
        for emission in range(self.num_emissions):
            KL_emission.append(KL_Dirichlet(self.emission_counts[:, emission, :],
                                            self.prior_emission_counts[:, emission, :]))

        # Equation 3.79 of Beal thesis
        ELBO = - np.sum(KL_start) - np.sum(KL_transition) - np.sum(KL_emission) + np.sum(np.log(scaling))
        return ELBO

    def M_step(self, data, expected_latent_state, expected_latent_state_pair):
        """
        Update the start, transition and emission posteriors by adding sufficient statistics of the hidden state
        posterior to the parameter priors.
        :param data: Training data. Shape = (number of emissions X number of sequences X length of sequence)
        :param expected_latent_state: Posterior probability of a single state at a time point.
                 Shape (num_states, shape of data).
        :param expected_latent_state_pair: Posterior probability of pairs of states.
                 Shape (num_states, num_states, shape of data).
        :return:
        """
        # Equations 3.54 to 3.59 in Beal thesis
        self.start_counts = self.prior_start_counts + np.sum(expected_latent_state[:, :, 0], axis = 1)
        self.transition_counts = self.prior_transition_counts + np.sum(expected_latent_state_pair, axis = (2, 3))
        self.emission_counts = np.zeros((self.num_states, self.num_emissions, 2))
        self.emission_counts += self.prior_emission_counts
        self.emission_counts[:, :, 0] += np.sum(expected_latent_state[:, None, :, :] * data[None, :, :, :],
                                                axis = (2, 3))
        self.emission_counts[:, :, 1] += np.sum(expected_latent_state[:, None, :, :] * (1.0 - data[None, :, :, :]),
                                                axis=(2, 3))

    def update_hyperpriors(self):
        """
        Optimize hyperpriors by minimizing KL divergence between posterior and prior distributions of parameters.
        Uses the minimize_KL_Bernoulli helper function.
        :return:
        """
        optimizer = minimize(minimize_KL_Bernoulli, np.array([self.start_hyperprior, self.transition_hyperprior,
                                                    self.emission_hyperprior]), args = (self,),
                             constraints = {"type": "ineq", "fun": lambda x: x})
        self.start_hyperprior = optimizer["x"][0]
        self.transition_hyperprior = optimizer["x"][1]
        self.emission_hyperprior = optimizer["x"][2]

    def fit(self, data, transition_hyperprior, emission_hyperprior, start_hyperprior, initial_transition_counts,
            initial_emission_counts, initial_start_counts, update_hyperpriors = True, update_hyperpriors_iter = 1,
            verbose = True):
        """
        Run the variational inference algorithm in Beal (2003). Assumes symmetric priors on start, transition and
        emission parameters with hyperpriors that are optimized as well.
        :param data: Training data. Shape = (number of emissions X number of sequences X length of sequence)
        :param transition_hyperprior: Hyperprior of transition counts
        :param emission_hyperprior: Hyperprior of emission counts
        :param start_hyperprior: Hyperprior of start counts
        :param initial_transition_counts: Initial guess of posterior transition counts.
               Shape = (num_states X num_states)
        :param initial_emission_counts: Initial guess of posterior emission counts. Shape = (num_states X num_emissions X 2)
        :param initial_start_counts: Initial guess of posterior start counts. Shape = (num_states)
        :param update_hyperpriors: Whether hyperpriors should be updated. Defaults to True.
        :param update_hyperpriors_iter: Number of iterations between hyperprior update steps. Defaults to 1.
        :param verbose: Show the improvement in ELBO through training. Default = True
        :return:
        """
        self.transition_hyperprior = transition_hyperprior
        self.emission_hyperprior = emission_hyperprior
        self.start_hyperprior = start_hyperprior
        self.transition_counts = initial_transition_counts
        self.emission_counts = initial_emission_counts
        self.start_counts = initial_start_counts

        self.converged = False

        for iter in range(self.max_iter):
            self.prior_transition_counts = self.transition_hyperprior * np.ones((self.num_states, self.num_states))
            self.prior_emission_counts = self.emission_hyperprior * np.ones((self.num_states, self.num_emissions, 2))
            self.prior_start_counts = self.start_hyperprior * np.ones(self.num_states)

            self.get_subnormalized_probabilities()
            self.get_state_likelihood(data.astype("int"))
            alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = self.E_step()
            self.ELBO.append(self.get_ELBO(scaling))

            if iter >= 1:
                improvement = self.ELBO[-1] - self.ELBO[-2]
                if verbose:
                    print("Training improvement in ELBO:", improvement)
                if improvement <= self.threshold:
                    self.converged = True
                    break
                if update_hyperpriors:
                    if iter % update_hyperpriors_iter == 0:
                        self.update_hyperpriors()

            self.M_step(data.astype("int"), expected_latent_state, expected_latent_state_pair)

