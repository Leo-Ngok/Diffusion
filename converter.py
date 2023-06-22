from jittor.utils.pytorch_converter import convert

IN_FILENAME = 'unet_torch.py'
OUT_FILENAME = 'unet_jittor_out.py'
jit_code = ''

def main():
    with open(IN_FILENAME, 'r') as f:
        txt = f.read()
        jit_code = convert(txt)

    with open(OUT_FILENAME, 'w') as f:
        f.write(jit_code)

if __name__ == '__main__':
    main()