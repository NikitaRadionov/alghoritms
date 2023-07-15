def compute_crc8(bytes: list) -> int:
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
