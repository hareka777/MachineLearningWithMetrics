using Microsoft.ML.Data;

namespace MachineLearningWithMetrics.MLdotNET.DataModel.BankNotes
{
    //Source:https://machinelearningmastery.com/standard-machine-learning-datasets/
    public class BankNotesInput
    {
        //Variance of Wavelet Transformed image
        [LoadColumn(0)]
        public float Variance { get; set; }

        //Skewness of Wavelet Transformed image
        [LoadColumn(1)]
        public float Skewness { get; set; }

        //Kurtosis of Wavelet Transformed imageC:\Users\dell\source\repos\Szakdolgozat\MachineLearningWithMetrics\MLdotNET\DataModel\BankNotes\BankNotesInput.cs
        [LoadColumn(2)]
        public float Kurtosis { get; set; }

        //Entropy of image
        [LoadColumn(3)]
        public float Entropy { get; set; }

        //Class
        [LoadColumn(4)]
        public bool Label { get; set; }
    }
}
