# MSR_Winter_Project

## Proposal
### Objective
Implement a neural network that is capable of playing [Pokémon Red/Blue](https://en.wikipedia.org/wiki/Pokémon_Red_and_Blue "Pokémon Red/Blue") on the Nintendo GameBoy/GameBoy Color/GameBoy Advance emulator project known as [VisualBoyAdvance-M (VBA-M)](https://github.com/visualboyadvance-m/visualboyadvance-m "VisualBoyAdvance-M (VBA-M)").

### Overall Strategy
* Implement the neural network through [TFLearn](http://tflearn.org/ "TFLearn") and [TensorFlow](https://www.tensorflow.org/ "TensorFlow").
* Feed [VisualBoyAdvance-M (VBA-M)](https://github.com/visualboyadvance-m/visualboyadvance-m "VisualBoyAdvance-M (VBA-M)") Screen into that neural network.
* Feed the neural network into [VisualBoyAdvance-M (VBA-M)](https://github.com/visualboyadvance-m/visualboyadvance-m "VisualBoyAdvance-M (VBA-M)") as game controls.

### Neural Network Strategy
* Possible layers that will be incorporated are [Long Short Term Memory (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory "Long Short Term Memory (LSTM)"), [Recurrent Neural Network (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network "Recurrent Neural Network (RNN)"), [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network "Convolutional Neural Network (CNN)"), etc.
* Possible training through [Q-learning](https://en.wikipedia.org/wiki/Q-learning "Q-learning") and [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning "reinforcement learning").
* Possible optimization goals may include score, wins, badges, Pokémon level, etc.

### Stretch Goals
* Expand the neural network to use multiple emulators at the same time.
* Expand the neural network to use other role-playing ROMs.
* Construct a research paper out of my research that can be self published on my website.

### Fall Back Strategy
* Choose a subset of levels.
* Choose a different game (Puzzle-Bobble Millennium).

### [Pokémon Red/Blue](https://en.wikipedia.org/wiki/Pokémon_Red_and_Blue "Pokémon Red/Blue")
> “The player controls the protagonist from an [overhead perspective](https://en.wikipedia.org/wiki/Top-down_perspective "overhead perspective") and navigates him throughout the fictional region of Kanto in a quest to master [Pokémon battling](https://en.wikipedia.org/wiki/Pokémon_battle "Pokémon battling"). The goal of the games is to become the champion of the Pokémon League by defeating the eight [Gym Leaders](https://en.wikipedia.org/wiki/Gym_Leaders "Gym Leaders") and then the top four Pokémon trainers in the land, the [Elite Four](https://en.wikipedia.org/wiki/Elite_Four "Elite Foure"). Another objective is to complete the [Pokédex](https://en.wikipedia.org/wiki/Pokédex "Pokédex"), an in-game encyclopedia, by obtaining the 150 available Pokémon. Red and Blue utilize the [Game Link Cable](https://en.wikipedia.org/wiki/Game_Link_Cable "Game Link Cable"), which connects two games together and allows Pokémon to be traded or battled between games. Both titles are independent of each other but feature the same plot, and while they can be played separately, it is necessary for players to trade among both games in order to obtain all of the first 150 Pokémon.”

### [VisualBoyAdvance-M (VBA-M)](https://github.com/visualboyadvance-m/visualboyadvance-m "VisualBoyAdvance-M (VBA-M)")
> “Our goal is to improve upon VisualBoyAdvance by integrating the best features from the various builds floating around.”

### [TensorFlow](https://www.tensorflow.org/ "TensorFlow")
> “TensorFlow™ is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API.”

### [TFLearn](http://tflearn.org/ "TFLearn")
> “TFlearn is a modular and transparent deep learning library built on top of Tensorflow. It was designed to provide a higher-level API to TensorFlow in order to facilitate and speed-up experimentations, while remaining fully transparent and compatible with it.”

### [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network "Convolutional Neural Network (CNN)")
> “In [machine learning](https://en.wikipedia.org/wiki/Machine_learning "machine learning"), a **convolutional neural network (CNN, or ConvNet)** is a type of [feed-forward artificial neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network "feed-forward artificial neural network") in which the connectivity pattern between its [neurons](https://en.wikipedia.org/wiki/Artificial_neuron "neurons")  is inspired by the organization of the animal [visual cortex](https://en.wikipedia.org/wiki/Visual_cortex "visual cortex"). Individual cortical neurons respond to stimuli in a restricted region of space known as the receptive field. The receptive fields of different neurons partially overlap such that they tile the [visual field](https://en.wikipedia.org/wiki/Visual_field "visual field"). The response of an individual neuron to stimuli within its receptive field can be approximated mathematically by a [convolution](https://en.wikipedia.org/wiki/Convolution "convolution") operation.”

### [Recurrent Neural Network (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network "Recurrent Neural Network (RNN)")
> “A **recurrent neural network (RNN)** is a class of [artificial neural network](https://en.wikipedia.org/wiki/Artificial_neural_network "artificial neural network") where connections between units form a [directed cycle](https://en.wikipedia.org/wiki/Directed_cycle "directed cycle"). This creates an internal state of the network which allows it to exhibit dynamic temporal behavior. Unlike [feedforward neural networks](https://en.wikipedia.org/wiki/Feedforward_neural_networks "feedforward neural networks"), RNNs can use their internal memory to process arbitrary sequences of inputs.”

### [Long Short Term Memory (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory "Long Short Term Memory (LSTM)")
> “Like most RNNs, an LSTM network is [universal](https://en.wikipedia.org/wiki/Turing_completeness "universl") in the sense that given enough network units it can compute anything a conventional computer can compute, provided it has the proper [weight](https://en.wikipedia.org/wiki/Weight "weight") [matrix](https://en.wikipedia.org/wiki/Matrix_(mathematics) "matrix"), which may be viewed as its program. Unlike traditional RNNs, an LSTM network is well-suited to learn from experience to [classify](https://en.wikipedia.org/wiki/Classification_in_machine_learning "classify"), [process](https://en.wikipedia.org/wiki/Computer_data_processing "process") and [predict](https://en.wikipedia.org/wiki/Predict "predict") [time series](https://en.wikipedia.org/wiki/Time_series "time series") when there are very long time lags of unknown size between important events.”

### [Q-learning](https://en.wikipedia.org/wiki/Q-learning "Q-learning")
> “**Q-learning** is a model-free [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning "reinforcement learning") technique. Specifically, Q-learning can be used to find an optimal action-selection policy for any given (finite) [Markov decision process](https://en.wikipedia.org/wiki/Markov_decision_process "Markov decision process") (MDP). It works by learning an [action-value function](https://en.wikipedia.org/w/index.php?title=Action-value_function "action-value function") that ultimately gives the expected utility of taking a given action in a given state and following the optimal policy thereafter.”

**Weekly** | **Steps**
--- | ---
**Week 1** | **Week 5**
~~1. Add Visual Boy Advance to Project~~ | 29. Research reinforcement learning
~~2. Add Pokemon Red/Blue ROM to Project~~ | 30. Incorporate reinforcement learning to the training model
~~3. Add Back-up game (Puzzle Bobble) ROM to Project~~ | 31. Estimate time needed for experiment to execute 
~~4. Create GitHub~~ | (if the amount of time needed for the experiment to execute is longer than practical, reevaluate experiment to reduce the estimated time)
~~5. Create File for AI in Python~~ | 32. Debug code with the smallest ranges possible to save time
~~6. Create .gitignore~~ | 33. Begin execution of experiment
~~7. Install Ubuntu 16.04 on Test laptop~~ | 
 | **Week 6**
**Week 2** | 34. Evaluate findings of experiment
~~8. Create Readme for GitHub~~ | 35. First outline of paper
~~9. Send Screen of VBA-M as input to AI~~ | 36. Add experiment to outline
~~10. Send commands from AI to VBA-M~~ | 37. Add findings to outline
11. List Dependencies | 38. Final research if needed
12. Incorporate debug log file output | 39. Add research to outline
13. Record emulator screen and audio for debugging | 40. Convert outline to rough draft
14. Create Launch File | 41. Explain choices made for training parameters
 | 42. Compare findings to research
**Week 3** | 
15. Add graph/chart output | **Week 7**
16. Research CNN layers | 43. Proofread/fix rough draft
17. Create experiment with CNN layer(s) | 44. Add formatting & illustrations to paper
18. Research LSTM RNN layer(s) | 45. Have someone Northwestern that is knowledgeable on NNs proofread and/or edit the latest rough draft
19. Add LSTM RNN layer(s) to experiment | 46. Finalize research paper
20. Research fully connected layers | 47. Self publish research paper to website
21. Add fully connected layer(s) to experiment | 
 | 
**Week 4** | 
22. Research dropout layers | 
23. Add dropout layer(s) to experiment | 
24. Research Q-Learning | 
25. Incorporate training with Q-Learning | 
26. Identify and research hyper-parameters of the NN | 
27. Incorporate hyper-parameters of the NN | 
28. Add goal maximization to the experiment | 
