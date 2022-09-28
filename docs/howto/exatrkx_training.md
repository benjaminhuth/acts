# How to run Exa.TrkX training

This how-to presents a workflow for training the Exa.TrkX pipeline with ACTS data and then run inference with the generated models.

## Training data generation

We use the CSV writers of the ACTS examples framework to produce the training data. The following writers are necessary:

* `ActsExamples::CsvTrackingGeometryWriter` for basic detector information
* `ActsExamples::CsvParticleWriter` for truth particle information
* `ActsExamples::CsvSimHitWriter` for truth hit information
* `ActsExamples::CsvMeasurementWriter` for digitized measurements (optional, the pipelinecan also be trained with truth data)

You should obtain a directory structure similar to the following:

```text
data
|- detectors.csv
|- train_all
    |- event000000000-cells.csv
    |- event000000000-measurements.csv
    |- event000000000-measurement-simhit-map.csv
    |- event000000000-particles.csv
    |- event000000000-truth.csv
    |- event000000001-cells.csv
    |- ...
```

## Setting up the training

A simple tools to configure and run the training pipeline is [traintrack](https://github.com/murnanedaniel/train-track/tree/master/traintrack) by Daniel Murnane.

Click [here](exatrkx_hyperparameters) to come a reference of all hyperparameters.
