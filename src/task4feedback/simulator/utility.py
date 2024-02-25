
units = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12} 

def parse_size(size_str: str):
    number, unit = [string.strip() for string in size_str.split()]
    return int(float(number) * units[unit])


def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac
