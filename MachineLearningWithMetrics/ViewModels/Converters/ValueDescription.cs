using System;

namespace MachineLearningWithMetrics.ViewModels.Converters
{
    /*
     * Helper class for enum converter
     * sources: https://stackoverflow.com/questions/20290842/converter-to-show-description-of-an-enum-and-convert-back-to-enum-value-on-sele
     * https://stackoverflow.com/questions/6145888/how-to-bind-an-enum-to-a-combobox-control-in-wpf
     */
    public class ValueDescription
    {
        public Enum Value { get; internal set; }
        public string Description { get; internal set; }
    }
}