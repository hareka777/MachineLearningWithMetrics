# Welcome to my BSc Thesis!

During my BSc Thesis in 2019, I implemented different machine learning models and compared their performance using performance metrics.
This repository contains the application I created in order to complete my BSc Thesis.

## Summary

In this thesis, I compare different machine learning models’ performance with
metrics in case of different hyperparameters. The models are based on three
different data sources to process. Prior to the execution of the training and the evaluation, several
hyperparameters can be set by the user. The user is able to set the training algorithm, data source and the rate of the test data and the
whole data set. The test data rate specifies the training data rate as well.

When all the parameters are set by the user, we are ready to execute the selected model and evaluate its performance. (Please note, the trained model are saved and at this point, we only need to load them.) We save the execution results into a time-series
database using metrics. Next, we display all this real-time data aggregated on a
dashboard.

We measure different performance metrics, for example, the duration of training or
evaluation, the accuracy and the related resource usage, like CPU and
memory usage.

As the result of this thesis, we will be familiar with the applied technologies’
performance, key features, ,strengths, weaknesses and efficiency. This knowledge can support further projects
related to this subject.

The thesis primarily uses the ML.NET and the AppMetrics packages.

## Installation guide and steps

1.  Clone this repository.
2.  To set up Grafana, download it at https://grafana.com/grafana/download.
3.  Run the downloaded application.
4.  By default, Grafan is available at http://localhost:3000/.
5.  Download InfluxDB at https://portal.influxdata.com/downloads/.
6.  Run influxd.exe.
7.  Downloading Chronograf is optional, it is an administration interface for InfluxDB:
8.  Download Chronograf at https://portal.influxdata.com/downloads/.
9.  Run Chronograf, basically it is available at http://localhost:8888/.
10. To import Grafana dashboard, open Grafana at https://localhost:3000/ , open dashboard search and hit import.
    It is possible to upload the json file or paste it.
    This json file is available at Grafana folder in this repository.
    More information available at https://grafana.com/docs/grafana/latest/reference/export_import/.
11. Now, everything is ready to run and use the C# application.
   
Application header image source:
http://csreports.aspeninstitute.org/Roundtable-on-Artificial-Intelligence/2019/what-is

## The application

The following image shows a small part of the metrics dashboard that monitors the machine learning models' performance.

![image](https://user-images.githubusercontent.com/37445999/141823060-7748ff38-618a-4f20-8491-f1fb64448a6e.png)
