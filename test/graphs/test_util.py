def level_condition(a, n, m, p, l):
        return 3 * a * n * n > m[l] * p[l]

def calc_num_levels(a, n, m, p):
    l = 0
    try:
        while(l < len(m)):
        # while(not flag):
            if(level_condition(a, n, m, p, l)):
                l = l + 1
                # flag = False
            else:
                return l + 1
    except IndexError:
            return -1

def get_total_p_hier_mesh(levels, p_per_level):
    p = []
    total_p = 0
    for i in range(levels):
        p.append(p_per_level)
        total_p += int(pow(p_per_level, i + 1))
    return p, total_p