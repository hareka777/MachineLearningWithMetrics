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

        [ColumnName("DiffPrev11")]
        [LoadColumn(10)]
        public float DiffPrev11;

        [ColumnName("DiffPrev12")]
        [LoadColumn(11)]
        public float DiffPrev12;

        [ColumnName("DiffPrev13")]
        [LoadColumn(12)]
        public float DiffPrev13;

        [ColumnName("DiffPrev14")]
        [LoadColumn(13)]
        public float DiffPrev14;

        [ColumnName("DiffPrev15")]
        [LoadColumn(14)]
        public float DiffPrev15;

        [ColumnName("DiffPrev16")]
        [LoadColumn(15)]
        public float DiffPrev16;

        [ColumnName("DiffPrev17")]
        [LoadColumn(16)]
        public float DiffPrev17;

        [ColumnName("DiffPrev18")]
        [LoadColumn(17)]
        public float DiffPrev18;

        [ColumnName("DiffPrev19")]
        [LoadColumn(18)]
        public float DiffPrev19;

        [ColumnName("DiffPrev20")]
        [LoadColumn(19)]
        public float DiffPrev20;

        [ColumnName("DiffPrev21")]
        [LoadColumn(20)]
        public float DiffPrev21;

        [ColumnName("DiffPrev22")]
        [LoadColumn(21)]
        public float DiffPrev22;

        [ColumnName("DiffPrev23")]
        [LoadColumn(22)]
        public float DiffPrev23;

        [ColumnName("DiffPrev24")]
        [LoadColumn(23)]
        public float DiffPrev24;

        [ColumnName("Value")]
        [LoadColumn(24)]
        public float Value;

    }


}





