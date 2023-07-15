def compute_crc8(bytes: list) -> int:
    """
    Вычисление crc8 для последовательности байт
    Если значение crc8 добавить в конец входного списка bytes, то вычисленное значение crc8 для такой последовательности
    байт будет равно 0
    Args:
        bytes (list): Список из целых чисел, вмещающихся в 1 байт
    Returns:
        int: Значение crc8 для данной последовательности байт
    """
    generator = 0x11D # поменяй эту штуку под свою задачу
    crctable = [0] * 256

    for dividend in range(256):
        currByte = dividend

        for bit in range(8):
            if (currByte & 0x80) != 0:
                currByte <<= 1
                currByte ^= generator
            else:
                currByte <<= 1

        crctable[dividend] = currByte

    crc = 0
    for b in bytes:
        data = b ^ crc
        crc = crctable[data]
    return crc


def compute_crc8_simple(bytes):
    generator = 0x11D
    crc = 0

    for currByte in bytes:
        crc ^= currByte

        for i in range(8):
            if (crc & 0x80) != 0:
                crc = (crc << 1) ^ generator
            else:
                crc <<= 1

    return crc


def urlb64decode(b:bytes) -> list:
    """
    Декодирование байтовой строки из URLbase64 в последовательность байтов
    Args:
        b (bytes): Последовательность байт закодированная в URLbase64
                Выглядит примерно так: b'IgP_fwgDAghTV0lUQ0gwMQMFREVWMDEFREVWMDIFREVWMDMo'

    Returns:
        list: Массив чисел int представляющих десятичное значение байтов (по 8 бит)
    """
    b = str(b)
    b = b[2:len(b) - 1]
    table = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,
        'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
        'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14,
        'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19,
        'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24,
        'Z': 25, 'a': 26, 'b': 27, 'c': 28, 'd': 29,
        'e': 30, 'f': 31, 'g': 32, 'h': 33, 'i': 34,
        'j': 35, 'k': 36, 'l': 37, 'm': 38, 'n': 39,
        'o': 40, 'p': 41, 'q': 42, 'r': 43, 's': 44,
        't': 45, 'u': 46, 'v': 47, 'w': 48, 'x': 49,
        'y': 50, 'z': 51, '0': 52, '1': 53, '2': 54,
        '3': 55, '4': 56, '5': 57, '6': 58, '7': 59,
        '8': 60, '9': 61, '-': 62, '_': 63,
    }
    bits = []
    for i in range(len(b)):
        bb = bin(table[b[i]])[2:]
        if len(bb) == 6:
            bits.append(bb)
        else:
            bits.append('0'*(6 - len(bb)) + bb)
    bits = "".join(bits)
    cbytes = []
    for i in range(0, len(bits), 8):
        if i + 8 < len(bits):
            cbytes.append(int(bits[i:i+8], 2))
        else:
            mod = bits[i:]
            if len(mod) == 8: # последняя часть учитывается только если она является целым байтом (8 бит)
                cbytes.append(int(bits[i:], 2))
    return cbytes

def uleb128_encode(i: int) -> list:
    """
    Кодирование чисел в формате uleb128

    Args:
        i (int): обычное число

    Returns:
        list: список целых чисел представляющих собой
              последовательность байт, которыми кодируется
              переданное число
    """
    r = []
    length = 0
    if i == 0:
        r = [0]
        return r

    while i > 0:
        r.append(0)
        r[length] = i & 0x7F
        i >>= 7
        if i != 0:
            r[length] |= 0x80
        length += 1

    return r
