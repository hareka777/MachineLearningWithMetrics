using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;

namespace MachineLearningWithMetrics.ViewModels.Converters
{
    //source: https://stackoverflow.com/questions/20290842/converter-to-show-description-of-an-enum-and-convert-back-to-enum-value-on-sele
    //https://stackoverflow.com/questions/6145888/how-to-bind-an-enum-to-a-combobox-control-in-wpf

    /*
     * Helper class for enum converter
     */
    public static class EnumHelper
    {
        public static string Description(this Enum value)
        {
            var attributes = value.GetType().GetField(value.ToString()).GetCustomAttributes(typeof(DescriptionAttribute), false);
            if (attributes.Any())
                return (attributes.First() as DescriptionAttribute).Description;
            else
            {
                return "";
            }
        }

        public static IEnumerable<ValueDescription> GetAllValuesAndDescriptions(Type t)
        {
            if (!t.IsEnum)
                throw new ArgumentException($"{nameof(t)} must be an enum type");

            return Enum.GetValues(t).Cast<Enum>().Select((e) => new ValueDescription() { Value = e, Description = e.Description() }).ToList();
        }
    }
}
