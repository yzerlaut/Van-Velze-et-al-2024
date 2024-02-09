# Code for the Study: Van-Velze-et-al-2024

    **Feedforward circuit explains modality-specific control of cortical inhibition**

## Install

To fetch and be able to run the code of the study:
```
git clone https://github.com/yzerlaut/Van-Velze-et-al-2014 --recurse-submodules
```
N.B. to have it working you need the `--recurse-submodules` option to fetch the dependencies

## Spiking Network Model

1) We build the model by combining a feedforward and a disinhibitory pathway 
    into a model of recurrent cortical dynamics:
        
        [Building-the-Circuit-Model.ipynb](Building-the-Circuit-Model.ipynb)

2) We model the different behavioral events as a set of time-varying 
    neuromodulatory and sensory stimulation levels

        [Modulation-Props.ipynb](Modulation-Props.ipynb)

3) We run the resulting model:

        [Running-Model.ipynb](Running-Model.ipynb) 

## Data Analysis

we generate some of the plots of the preprocessed data in:

- [Pupil-Whisking-Running.ipynb](./Pupil-Whisking-Running.ipynb)
    we analyze the arousal level (with the Pupil proxy) across behavioral states
        (i.e. rest, whisking only, whisking+locomotion)

- [LMI-in-CNO-vs-CTRL.ipynb](./LMI-in-CNO-vs-CTRL.ipynb)
    we analyze the additional control of the CNO injection without the dreadds

- [Loading-Data.ipynb](./Loading-Data.ipynb)
    loads the `Analyzed-data.pickle`  files and fetch information from it



