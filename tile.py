import tflite
import argparse

class Transformer:
  def __init__(self, orig_model):
    self.orig_model = orig_model
    self.gen_buffers = []
    self.gen_opcodes = []
    self.gen_operators = [] # this should be 2d array axis 0 is for graph, axis 1 is for operators in graph
    self.gen_tensors = [] # same for tensors
    self.tensors_orig_to_gen = {}
    self.operators_orig_to_gen = {}
    self.builder = flatbuffers.Builder(1024)

  def process_graph(self, orig_graph):
    pass

  def finalize_opcodes(self):
    pass

  def process_all(self):
    version = self.orig_model.Version()
    
    description_string = new_builder.CreateString(sample_model.Description())
    for graph_idx in range(orig_model.SubgraphsLength()):
      process_graph(orig_model.Subgraphs(graph_idx))
    finalize_opcodes()

    # generate operator codes here
    # generate subgraphs
    # generate buffers vector
    buffers_num = len(self.gen_buffers)
    tflite.ModelStartBuffersVector(self.builder, buffers_num)
    for idx in reversed(range(buffers_num)):

      self.builder.PrependUOffsetTRelative(gen_buffer[idx])
    self.builder.EndVector(len(self.gen_buffers))

    tflite.ModelStart(self.builder)
    tflite.ModelAddVersion(self.buider, version)
    tflite.ModelAddOperatorCodes(self.builder, operator_codes)
    tflite.ModelAddSubgraphs(self.builder, subgraphs)
    tflite.ModelAddDescription(self.builder, description_string)
    tflite.ModelAddBuffers(self.builder, buffers)
    gen_model = tflite.ModelEnd(self.builder)

    self.builder.Finish(gen_model, file_identifier=b'TFL3')


  def gen_model(self):
    return self.builder.Output()
    

def main(args):
  orig_model = tflite.Model.GetRootAsModel(bytearray(args.input_model_file.read()))
  t = Transformer(orig_model)
  t.process_all()
  gen_model = t.gen_model()
  args.output_model_file.write(gen_model)

if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("input_model_file", type=argparse.FileType('rb'), help="path to input model")
  arg_parser.add_argument("output_model_file", type=argparse.FileType('wb'), help="path to output model")
  args = arg_parser.parse_args()
  main(args)

