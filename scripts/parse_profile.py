from collections import Counter
import fire

def parse_profile(fname):
    ms_total = Counter()
    for idx, line in enumerate(open(fname, 'r').readlines()):
        line = line.replace('|', '')
        line = line.replace('│', '')
        line = line.replace('├', '')
        line = line.replace('└', '')
        line = line.replace('─', '')
        parts = line.split()

        if len(parts) != 5:
            continue

        num, unit = parts[2][:-2], parts[2][-2:]
        num = float(num)
        if unit == 'us':
            num *= 0.001

        ms_total[parts[0]] += num

    total = sum(ms_total.values())
    for key, val in ms_total.items():
        print(f'{key:30} {val:5.2f} ms {val / total * 100:.2f}%')

    print(f'Total: {total:.2f} ms')

if __name__ == "__main__":
    fire.Fire(parse_profile)
