# Jupyter Notebooks on Machine Learning

I implement some interesting Machine Learning topics in Jupyter Notebooks, mostly from scratch. The focus is on understanding rather than achieving the state-of-the-art results. So far I have implemented Gaussian Process, Variational Auto Encoder (VAE), Natural Gradient and Learning Trigonometric Functions using Neural Networks. I hope these help you understand the topics better.

1. `Gaussian Process.ipynb`: Gaussian Process.
	<p float="center">
		<img src="figures/gp_prior.png" width="300"/>
		<img src="figures/gp.png" width="300"/>
	</p>

1. `VAE.ipynb`: Variational Autoencoder for MNIST.
	<p float="center">
		<img src="figures/VAE_latent.png" width="250"/>
		<img src="figures/VAE_latent_dec.png" width="250"/>
	    <img src="figures/VAE_samples.png" width="250"/>
	</p>

1. `Natural Gradient.ipynb`: Natural gradient to learn the parameters of a 1D Gaussian. Empirically I observe that natural gradient ascent converges faster than simple gradient ascent. Moreover, the KL divergence between 2 likelihood functions at consecutive training steps remains roughly the same during training. For theory on Natural Gradient I recommend reading [Agustinus Kristladl's Blog](https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/#:~:text=Up%20to%20constant%20factor%20of,%E2%88%87%CE%B8L(%CE%B8).)
	<p float="center">
	    <img src="figures/natural_contour.png" width=250/>
	    <img src="figures/natural_gradient_field.png" width=200/>
	    <img src="figures/natural_likelihood_map.png" width=250/>
	</p>

1. `Bayesian Linear Regression.ipynb`: Bayesian linear regression for 1D data using Gaussian form for both prior and likelihood, and known variance of likelihood term. The confidence of our model increases with more data points. This really demonstrates the power of Bayesian learning - when we have less data model itself tells us that it is less confident!
	<p float="center">
		<img src="figures/n=3, Prior.png" width="200">		
		<img src="figures/n=3, Posterior.png" width="200">
		<img src="figures/n=20, Posterior.png" width="200">
		<img src="figures/n=100, Posterior.png" width="200">
	</p>

1. `learn_transform.py`: Can a neural network learn forward and inverse trigonometric (sine/cosine/tangent) functions? My conclusion is that it can learn it in and around the regions where it has seen the data. Sigmoid activation works better than ReLU.
	<p float="center">
	    <img src="figures/sigmoid.png" width=300/>
	    <img src="figures/relu.png" width=300/>
	</p>




