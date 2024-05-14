

from deepseek_vl.models.processing_vlm import VLChatProcessor


model_path = "deepseek-ai/deepseek-vl-1.3b-base"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

decode_me = [100000,   2054,    418,    245,   9394,   4706,    285,  10046,  20308,
             13,   1257,    418,   2249,    276,   2579,    254,   7959,   3093,
            344,    254,   2677,   4614,     11,    285,   4750,    254,   2677,
            366,    245,   6265,    280,   9224,   1244,   3892,   4706,     13,
            185,    185,   5726,     25, 100015,  23853,     25,   1257,   7048,
            245,  20002,     13,  35567,    317,   9807,    276,   3471,    359,
            327,    363,     16,     15,     15,     11,     15,     15,     15,
            548,    340,    481,   2088,    327,    691,     13,   1724,   1266,
            744,    340,    837,    276,   2088,    327,    359,     30,    185,
            185,  15843,   1221,     25,    185,     32,    570,   1733,   1266,
            372,    304,    481,    752,    327,    359,     13,    185,     33,
            570,    304,    744,   2102,    359,    276,    768,    327,   2094,
             13,    185,    185,  32349,     25,    185,   7900,  10346,    520,
           3510,   5191,  27336,    366,    245,   2816,   5371,    185,    185,
          77398,     25,     32, 100001]
decoded = tokenizer.decode(decode_me)
print(decoded)
encode_me = "B"
encoded = tokenizer.encode(encode_me)
print(encoded)

