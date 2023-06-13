# MOON_TestSelection
# The replication package of MOON

This paper proposes MOON, a white-box test input selection approach based on multi-objective optimization. The neuron spectrum is proposed to
localize suspicious neurons that contribute to erroneous decisions made by DNN models. MOON formulates the test input selection method into a search-based testing problem. By tailoring a multiobjective optimization algorithm, it guides the search process towards maximizing the outputs of suspicious neurons while promoting diversity in neuron behaviors.

# 
craft_adv_examples.py: generate adversarial examples to simulate different testing contexts by using four adversarial attack strategies, i.e., FGSM, JSMA, Bim-a, and Bim-b.

locate_sus_neurons.py: locate suspicious neurons by conducting neuron spectrum analysis. 

retrain_iterate.py: retrain DNN models by adding five additional epochs.

run_nsga2.py, entropy.py, and EA.py: the multi-objective optimization algorithm for test input selection.
