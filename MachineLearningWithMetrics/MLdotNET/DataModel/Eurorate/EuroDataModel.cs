using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningWithMetrics.MLdotNET.DataModel.Eurorate
{
    public class EuroDataModel
    {

        /*[ColumnName("Value")]
        [LoadColumn(0)]
        public float Value;

        [ColumnName("PreviousDifferences")]
        [VectorType(64)]
        public float[] PreviousDifferences;*/

        

        [ColumnName("DiffPrev1")]
        [LoadColumn(0)]
        public float DiffPrev1;

        [ColumnName("DiffPrev2")]
        [LoadColumn(1)]
        public float DiffPrev2;

        [ColumnName("DiffPrev3")]
        [LoadColumn(2)]
        public float DiffPrev3;

        [ColumnName("DiffPrev4")]
        [LoadColumn(3)]
        public float DiffPrev4;

        [ColumnName("DiffPrev5")]
        [LoadColumn(4)]
        public float DiffPrev5;

        [ColumnName("DiffPrev6")]
        [LoadColumn(5)]
        public float DiffPrev6;

        [ColumnName("DiffPrev7")]
        [LoadColumn(6)]
        public float DiffPrev7;

        [ColumnName("DiffPrev8")]
        [LoadColumn(7)]
        public float DiffPrev8;

        [ColumnName("DiffPrev9")]
        [LoadColumn(8)]
        public float DiffPrev9;

        [ColumnName("DiffPrev10")]
        [LoadColumn(9)]
        public float DiffPrev10;

        [ColumnName("Value")]
        [LoadColumn(10)]
        public float Value;

    }


}





