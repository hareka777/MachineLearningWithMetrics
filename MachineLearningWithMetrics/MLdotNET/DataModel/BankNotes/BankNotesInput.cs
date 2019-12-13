using Microsoft.ML.Data;

namespace MachineLearningWithMetrics.MLdotNET.DataModel.BankNotes
{
    /*
     * Data Source: https://machinelearningmastery.com/standard-machine-learning-datasets/
     * Data model class for binary classification
     */
    public class BankNotesInput
    {
        //Variance of image
        [LoadColumn(0)]
        public float Variance { get; set; }

        //Skewness of image
        [LoadColumn(1)]
        public float Skewness { get; set; }

        //Kurtosis of image
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
