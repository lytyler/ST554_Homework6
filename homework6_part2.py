#import modules
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from sklearn import linear_model
import statistics as st

class SLR_slope_simulator:
    """
    a Python class that simulates the sampling distribution 
    of a slope estimator
    """
#create initial attributes of beta_0, beta_1, sigma, x, n, rng, and slopes (an empty list)
    n = None
    rng = None
    slopes = []

    def __init__(self, beta_0: float, beta_1: float, x, sigma: float, seed: int):
        """
        initialize the class using _ init _ with arguments self, beta_0, beta_1, 
        x, sigma, and seed
        """
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.x = x
        self.sigma = sigma
        self.seed = seed
        self.rng = np.random.default_rng(seed) #set seed

    def generate_data(self):
        "method that generates one dataset, returning x and y"
        SLR_slope_simulator = self
        for i in range(len(SLR_slope_simulator.x)):
            y = SLR_slope_simulator.beta_0 \
                + SLR_slope_simulator.beta_1*SLR_slope_simulator.x \
                + SLR_slope_simulator.rng.normal(0, SLR_slope_simulator.sigma, \
                                                 len(SLR_slope_simulator.x))
        return(SLR_slope_simulator.x, y)

    def fit_slope(self, x, y):
        """
        #method that takes in an x and y and fits the SLR model, returning the 
        estimated slope
        """
        SLR_slope_simulator = self
        reg = linear_model.LinearRegression() #create regression object
        fit = reg.fit(x.reshape(-1, 1),y) #fit model
        return(fit.coef_)

    def run_simulations(self, n):
        """
        method that takes in n (number of simulations) and populates the slopes 
        list with simulated slopes
        """
        SLR_slope_simulator = self
        for i in range(1, n):
            a, b = SLR_slope_simulator.generate_data()
            #print(SLR_slope_simulator.slopes)
            SLR_slope_simulator.slopes.append(SLR_slope_simulator.fit_slope(a, b)[0])

    def plot_sampling_distribution(self):
        "#method that plots a histogram of the list of simluated slopes"
        SLR_slope_simulator = self
        if len(SLR_slope_simulator.slopes) <= 0:
            print("Please execute .run_simulations() before executing .plot_sampling_distribution()")
        else:
            plt.hist(SLR_slope_simulator.slopes, bins = 20)
            plt.title("Histogram of Simulated Slope Values")
            plt.xlabel("Slope Values")
            plt.ylabel("Frequency")

#method that takes in a slope value and a sided argument and estimates probability 
    def find_prob(self, value, sided):
        """
        method that takes in a slope value and a sided argument and estimates 
        probability 
        """
        SLR_slope_simulator = self
        if len(SLR_slope_simulator.slopes) <= 0:
            print("Please execute .run_simulations() before executing .find_prob()")
            return
        if sided == "above":
            prob = round \
                (np.asarray(SLR_slope_simulator.slopes)[np.asarray(SLR_slope_simulator.slopes) > value] \
                .shape[0]/len(SLR_slope_simulator.slopes), 4)
        elif sided == "below":
            prob = round \
                (np.asarray(SLR_slope_simulator.slopes)[np.asarray(SLR_slope_simulator.slopes) < value] \
                .shape[0]/len(SLR_slope_simulator.slopes), 4)
        else:
            if value > st.median(SLR_slope_simulator.slopes):
                prob = round \
                (2*np.asarray(SLR_slope_simulator.slopes)[np.asarray(SLR_slope_simulator.slopes) > value] \
                .shape[0]/len(SLR_slope_simulator.slopes), 4)
            elif value < st.median(SLR_slope_simulator.slopes):
                prob = round \
                (2*np.asarray(SLR_slope_simulator.slopes)[np.asarray(SLR_slope_simulator.slopes) < value] \
                .shape[0]/len(SLR_slope_simulator.slopes), 4)
        return(prob)
    
#create an instance as specified
demo = SLR_slope_simulator(12.0, 2.0, 
                           np.array(list(np.linspace(start = 0, stop = 10, num = 11))*3), 
                           1.0, 10)

#call plot_sampling_distribution, error message should be returned
demo.plot_sampling_distribution()

#run 10,000 simulations
demo.run_simulations(10000)

#plot histogram of simulated sampling distribution
demo.plot_sampling_distribution()

#approximate probability as specified
demo.find_prob(2.1, "two-sided")

#print out simulated slope values using slopes attribute
demo.slopes