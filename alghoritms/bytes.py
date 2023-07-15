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
