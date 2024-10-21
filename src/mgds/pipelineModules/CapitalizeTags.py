from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
import string


class CapitalizeTags(
    PipelineModule,
    RandomAccessPipelineModule,
):
    def __init__(
            self,
            text_in_name: str,
            enabled_in_name: str,
            probability_in_name: float,         #from 0-1, probability capitalization adjustment will be applied to each tag
            capitalize_mode_in_name: str,       #comma-separated list of which caps changes can be applied, accepts "capslock, title, first, random"
            delimiter_in_name: str,
            convert_lowercase_in_name: str,     #whether to apply lowercase to entire caption before any other caps changes
            text_out_name: str,
    ):
        super(CapitalizeTags, self).__init__()
        self.text_in_name = text_in_name
        self.enabled_in_name = enabled_in_name
        self.probability_in_name = probability_in_name
        self.capitalize_mode_in_name = capitalize_mode_in_name
        self.delimiter_in_name = delimiter_in_name
        self.convert_lowercase_in_name = convert_lowercase_in_name
        self.text_out_name = text_out_name

    def length(self) -> int:
        return self._get_previous_length(self.text_in_name)

    def get_inputs(self) -> list[str]:
        return [self.text_in_name, self.enabled_in_name, self.probability_in_name, self.capitalize_mode_in_name, self.delimiter_in_name, self.convert_lowercase_in_name]

    def get_outputs(self) -> list[str]:
        return [self.text_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        text = self._get_previous_item(variation, self.text_in_name, index)
        delimiter = self._get_previous_item(variation, self.delimiter_in_name, index)
        enabled = self._get_previous_item(variation, self.enabled_in_name, index)
        probability = self._get_previous_item(variation, self.probability_in_name, index)
        capitalize_mode = self._get_previous_item(variation, self.capitalize_mode_in_name, index)
        convert_lowercase = self._get_previous_item(variation, self.convert_lowercase_in_name, index)
        rand = self._get_rand(variation, index)


        #if convert_lower is enabled always convert caption to lowercase before any other changes
        if convert_lowercase:
            text = text.lower()

        #randomly pick one of the capitialization methods for each tag
        if enabled and (probability > 0):
            text_list = []
            for s in [tag.strip() for tag in text.split(delimiter)]:
                if rand.random() < probability:
                    capmode_list = [tag.strip() for tag in capitalize_mode.split(",")]
                    match rand.choice(capmode_list):
                        case "capslock":  #make ALL CAPS
                            text_list.append(s.upper())
                        case "title":     #make First Letter Of All Words Caps
                            text_list.append(string.capwords(s))
                        case "first":     #make First word only caps
                            text_list.append(s.capitalize())
                        case "random":    #make rAnDOm lETTerS CaPs
                            s2 = ''.join(rand.choice(x) for x in zip(s.upper(), s.lower()))
                            text_list.append(s2)
                        case _:           #catch any unexpected values
                            text_list.append(s)
                else:
                    text_list.append(s)
            text = delimiter.join(text_list)
 
        return {
            self.text_out_name: text
        }
