import sys


### 残骸
# def train_kytea(train_path, model_path, dict_path):
#     cmd = 'train-kytea -full ' + train_path + ' -model ' + model_path
#     if dict_path:
#         cmd += ' -dict ' + dict_path
#     p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     out, err = p.communicate()
#     print(out.decode('utf-8').rstrip())
#     return


if __name__=='__main__':
    input = sys.argv[1]
    output = sys.argv[2]
    data.process_bccwj_data_for_kytea(input, output)
