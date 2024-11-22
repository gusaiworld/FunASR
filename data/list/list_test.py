import os
import re


def find_difference(item, list2):
    difference = [False if item not in list2 else True]
    if not difference:
        print(item+'?')
    return difference[0]

def build_list(
        find_type='.rttm',  #生成list的文件类型

        base1_dir='/data/guyf/data_dir/record',  #寻找文件所在
        base2_dir='/home/guyf/DiariZen/recipes/diar_ssl/data/AMI_AliMeeting_AISHELL4/test/recording/wav.scp',
        bar_type='/',  #斜杠方向
        cnt='tra',  #相对级数
        type_1=True,  #TRUE：绝对路径    FALSE：想对路径
        sensor=[0.000, 490.555]
):
    rec_dir = find_type[1:]
    filedir = base1_dir
    i=0
    print(filedir)
    expdir = base2_dir
    pattern = r"\w+(?= )"
    d = []
    if cnt == 'tra':
        dir1 = '/data/guyf/funasr/FunASR_sv/data/list/text_tra.txt'
    elif cnt == 'dev':
        dir1 = '/data/guyf/funasr/FunASR_sv/data/list/text_dev.txt'
    with open(dir1, 'r', encoding='utf-8') as f:
        wav_list = f.readlines()
    for i in range(len(wav_list)):
        matches = re.findall(pattern, wav_list[i])
        d.append(matches[0])
    if find_type == 'last':
        #当前.py文件路径 for rttmlist
        with open(expdir, 'w', encoding='utf-8') as stone:  # 建立文件
            for root, dirs, files in os.walk(filedir):  # 遍历
                for name in files:
                    if name.endswith('.wav'):
                        if type_1:  # 绝对路径
                            # stone.write('examples' + bar_type + name + '\n')  # 写路径
                            stone.write(name[:-4]+' '+'1'+' '+str(sensor[0]) +' '+ str(sensor[1])+'\n')  # 写路径
                    i+=1
                    print(i)

        wavs = []
        try:
            # input is wav list
            # with open('/home/guyf/3D-Speaker/3D-Speaker-main/egs/3dspeaker/speaker-diarization/examples/wav.list', 'r') as word_list:
            with open(expdir, 'r') as f:
                wav_list = f.readlines()

        except:
            raise Exception('Input should be a wav file or a wav list.')
    elif find_type == '.wav':
        #当前.py文件路径 for wavlist
        find_type = '.wav'  # 生成list的文件类型
        with open(expdir, 'w', encoding='utf-8') as stone:
            for root, dirs, files in os.walk(filedir):  #遍历
                for name in files:
                    if name.endswith(find_type):
                        if type_1:  #绝对路径
                            if os.path.getsize(root + bar_type + name)> 1000 and find_difference(name[:-4],d):
                                stone.write(name[:-4]+'  '+root + bar_type + name + '\n')  #写路径
                            else :
                                print(root + bar_type + name)
                        else:  #相对路径
                            stone.write(
                                root.replace(filedir + bar_type, ('..' + bar_type) * cnt) + bar_type + name + '\n')

        wavs = []

    try:
        # input is wav list
        # with open('/home/guyf/3D-Speaker/3D-Speaker-main/egs/3dspeaker/speaker-diarization/examples/wav.list', 'r') as word_list:
        with open(expdir, 'r') as f:
            wav_list = f.readlines()

    except:
        raise Exception('Input should be a wav file or a wav list.')


def main( ):

    build_list(find_type='.wav',  base1_dir='/data/guyf/funasr/FunASR/examples/aishell/raw_data/data_aishell/wav/train',
        base2_dir='/data/guyf/funasr/FunASR_sv/data/list/wav_tra.scp',cnt='tra',
       )
    build_list(find_type='.wav', base1_dir='/data/guyf/funasr/FunASR/examples/aishell/raw_data/data_aishell/wav/dev',
               base2_dir='/data/guyf/funasr/FunASR_sv/data/list/wav_dev.scp',cnt='dev',
               )


if __name__ == '__main__':
   main()
