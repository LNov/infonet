filename = 'LOG'
open(filename + '_checkstatus.txt', 'w').writelines(
    line for line in open(filename + '.txt') if ('#' in line or 'pypet' in line))