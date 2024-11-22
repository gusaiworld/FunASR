import os
import re
import argparse
parser = argparse.ArgumentParser(description='Generate wav.scp.')
parser.add_argument('--tra_file', default='', help='训练集位置')
parser.add_argument('--output_dir', default='',  help='输出wav.scp位置')
parser.add_argument('--txt_tra', default='', help='对比txt以免有多余文件')
parser.add_argument('--dev_file', default='',  help='dev')
parser.add_argument('--txt_dev', default='',  help='dev txt')
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
        txt_file='/data/guyf/funasr/FunASR_sv/data/list/text_tra.txt',  #相对级数
        type_1=True,  #TRUE：绝对路径    FALSE：想对路径
        rec_id='wav_tra',sensor=[0,1]#忽略
):
    rec_dir = find_type[1:]
    filedir = base1_dir
    i=0
    print(filedir)
    expdir = os.path.join(base2_dir, rec_id + '.scp')
    pattern = r"\w+(?= )"
    d = []
    dir1=txt_file
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


        wavs = []

    try:
        # input is wav list
        # with open('/home/guyf/3D-Speaker/3D-Speaker-main/egs/3dspeaker/speaker-diarization/examples/wav.list', 'r') as word_list:
        with open(expdir, 'r') as f:
            wav_list = f.readlines()

    except:
        raise Exception('Input should be a wav file or a wav list.')


def main( ):
    args = parser.parse_args()
    build_list(find_type='.wav',  base1_dir=args.tra_file,base2_dir= args.output_dir,txt_file=args.txt_tra,rec_id='wav_tra')
    build_list(find_type='.wav', base1_dir=args.dev_file, base2_dir=args.output_dir,txt_file=args.txt_dev,rec_id='wav_dev')


if __name__ == '__main__':
   main()
