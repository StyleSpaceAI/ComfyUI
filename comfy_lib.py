import os
import shutil
import torch
from nodes import init_custom_nodes, NODE_CLASS_MAPPINGS
import execution
import folder_paths
import nodes
import copy
import gc
import traceback

class PromptLib:
    def __init__(self):
        init_custom_nodes()

    def add_prompt(self, prompt, extra_data={}):
        e = SyncPromptExecutor()
        e.execute(prompt=prompt, extra_data=extra_data)
        return e.outputs

    def add_image(self, image_path):
        upload_dir = folder_paths.get_input_directory()

        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        filename = os.path.basename(image_path)
        shutil.copyfile(image_path, os.path.join(upload_dir, filename))

class SyncPromptExecutor:
    def __init__(self):
        self.outputs = {}
        self.old_prompt = {}

    def execute(self, prompt, extra_data={}):
        nodes.interrupt_processing(False)

        with torch.inference_mode():
            self.delete_changed_outputs(prompt)
            current_outputs = set(self.outputs.keys())
            executed = self.execute_output_nodes(prompt, extra_data, current_outputs)
            self.update_old_prompt(executed, prompt)

        self.cleanup_memory()

    def delete_changed_outputs(self, prompt):
        for x in prompt:
            execution.recursive_output_delete_if_changed(prompt, self.old_prompt, self.outputs, x)

    def execute_output_nodes(self, prompt, extra_data, current_outputs):
        executed = []
        try:
            to_execute = self.get_output_nodes_to_execute(prompt)

            while len(to_execute) > 0:
                to_execute = sorted(list(map(lambda a: (len(execution.recursive_will_execute(prompt, self.outputs, a[-1])), a[-1]), to_execute)))
                x = to_execute.pop(0)[-1]

                if self.is_valid_input(prompt, x):
                    executed += recursive_execute(prompt, self.outputs, x, extra_data)
        except Exception as e:
            print(traceback.format_exc())
            self.delete_new_outputs(current_outputs)

        return executed

    def get_output_nodes_to_execute(self, prompt):
        to_execute = []
        for x in prompt:
            class_ = nodes.NODE_CLASS_MAPPINGS[prompt[x]['class_type']]
            if hasattr(class_, 'OUTPUT_NODE'):
                to_execute += [(0, x)]
        return to_execute

    def is_valid_input(self, prompt, x):
        try:
            valid, _ = execution.validate_inputs(prompt, x)
            return valid
        except:
            return False

    def delete_new_outputs(self, current_outputs):
        to_delete = [o for o in self.outputs if o not in current_outputs]
        for o in to_delete:
            if o in self.old_prompt:
                del self.old_prompt[o]
            del self.outputs[o]

    def update_old_prompt(self, executed, prompt):
        executed = set(executed)
        for x in executed:
            self.old_prompt[x] = copy.deepcopy(prompt[x])

    def cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            if torch.version.cuda: #This seems to make things worse on ROCm so I only do it for cuda
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def recursive_execute(prompt, outputs, current_item, extra_data={}):
    unique_id = current_item
    inputs = prompt[unique_id]['inputs']
    class_type = prompt[unique_id]['class_type']
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
    if unique_id in outputs:
        return []

    executed = []

    for x in inputs:
        input_data = inputs[x]

        if isinstance(input_data, list):
            input_unique_id = input_data[0]
            output_index = input_data[1]
            if input_unique_id not in outputs:
                executed += recursive_execute(prompt, outputs, input_unique_id, extra_data)

    input_data_all = execution.get_input_data(inputs, class_def, unique_id, outputs, prompt, extra_data)
    obj = class_def()

    nodes.before_node_execution()
    outputs[unique_id] = getattr(obj, obj.FUNCTION)(**input_data_all)
    if "ui" in outputs[unique_id]:
        if "result" in outputs[unique_id]:
            outputs[unique_id] = outputs[unique_id]["result"]
    return executed + [unique_id]