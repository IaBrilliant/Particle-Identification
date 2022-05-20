# Particle-Identification

The project was completed in 2020 jointly with Benjamin Forster, supervised by Prof. Morgan Wascko.
Personal responsibilities included generation of samples and CNN development. 
 

Convolutional Neural Networks (CNNs) have been widely applied as a high-accuracy visual recognition tool 
to solve various classification problems. We studied the performance of a multi-layer CNN employed as a classifier
for a commonly used particle identification technique - Cherenkov Ring-imaging. 

<p align="center">
<img width="370" alt="Cherenkov" src="https://user-images.githubusercontent.com/66305897/169488702-37bf46d8-bb05-463c-a8e9-a5383db1d0b7.png">
<p align="center">
Cherenkov Rings (Adopted from Belle II Experiment)

The operating principle relies on measuring the Cherenkov angle when a charged particle traverses a dispersive medium. In order to generate patterns
that closely resemble raw experimental data, dimensional parameters of a real proximity focusing detector (ARICH)
were integrated into simulation. 
  
<p align="center">
<img width="843" alt="Pion:Kaon" src="https://user-images.githubusercontent.com/66305897/169489600-5bf13b14-3cad-4205-8322-cc4303e3940e.png">
<p align="center">
Simulated Pion and Kaon rings with the color scale representing the number of photon hits

The network was trained to classify three categories of patterns produced by pion,
kaon, and proton beams with 5 GeV/c momenta. The CNN performed exceptionally well when differentiating
between proton and kaon patterns, achieving 100% accuracy after a single epoch, however, it did not reach its
full potential when classifying pion and kaon patterns - particles with small angular difference. To improve the
efficiency we introduced changes to the network’s architecture, adding another convolutional layer with greater
number of filters.
 
<p align="center">
<img width="851" alt="Architecture" src="https://user-images.githubusercontent.com/66305897/169490498-1fc5caf1-ea92-451d-b584-ab3cf19b59bf.png">
<p align="center">
Architecture of the Convolutional Neural Network in use
  
<p align="center">
<img width="863" alt="Results" src="https://user-images.githubusercontent.com/66305897/169490925-98a61967-a323-49aa-8c22-c1c802adc0b3.png">
<p align="center">
Improved training and testing accuracy upon introduction of an additional layer
  
We found that this tuning produced a significant effect on the network’s accuracy and time
of execution. We discuss the results of this investigation as well as give recommendations for improvement of
simulation model.
  
