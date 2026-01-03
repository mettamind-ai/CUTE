# Rút gọn từ https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4neo/src/binidx.py và https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4neo/src/dataset.py
import torch, struct, random
import numpy as np
from functools import lru_cache

class Index(object):
    '''định nghĩa lớp Index để đọc dữ liệu từ file index. 
    Đầu tiên, lớp này định nghĩa một số hằng số, bao gồm:

    _HDR_MAGIC: chuỗi byte dùng để kiểm tra xem file index có đúng định dạng hay không.

    _DTYPES: một từ điển ánh xạ các số nguyên đại diện cho kiểu dữ liệu với các đối tượng kiểu dữ liệu tương ứng trong thư viện NumPy.
    '''
    _HDR_MAGIC = b"MMIDIDX\x00\x00"

    _DTYPES = {
        1: np.uint8,  2: np.int8,
        3: np.int16,  4: np.int32,
        5: np.int64,  6: np.single,
        7: np.double, 8: np.uint16,
    }

    def __init__(self, path):
        '''Phương thức __init__(self, path) là phương thức khởi tạo của lớp Index, nhận đầu vào là đường dẫn tới file index. Trong phương thức này, đầu tiên nó đọc nội dung của file index, kiểm tra xem định dạng file có đúng hay không, rồi lưu thông tin về kiểu dữ liệu, số lượng văn bản và số lượng từ (từ được định nghĩa bởi kích thước của mỗi mục trong file index) vào các thuộc tính của đối tượng.
        '''
        with open(path, "rb") as stream:
            magic_test = stream.read(9)
            assert self._HDR_MAGIC == magic_test, "not correct file format"

            self.version = struct.unpack("<Q", stream.read(8)) # đọc version
            (dtype_code,) = struct.unpack("<B", stream.read(1)) # đọc kiểu dữ liệu của file

            self._dtype = self._DTYPES[dtype_code]
            self._dtype_size = self._dtype().itemsize

            self._len = struct.unpack("<Q", stream.read(8))[0]
            self._doc_count = struct.unpack("<Q", stream.read(8))[0]
            offset = stream.tell() # trỏ tới dữ liệu của văn bản đầu tiên

        '''Sau đó, phương thức sử dụng thư viện NumPy để tạo các mảng lưu trữ kích thước và vị trí của các mục trong file index. Các mảng này được tạo bằng cách sử dụng đối tượng np.memmap để ánh xạ file index vào bộ nhớ, giúp cho việc truy cập dữ liệu trở nên nhanh chóng hơn.
        '''
        self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

        self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
        self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len, \
            offset=offset + self._sizes.nbytes)

    def __del__(self): # thu hồi bộ nhớ
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap

    def __len__(self):
        return self._len

    # @lru_cache(maxsize=256)
    def __getitem__(self, i):
        '''cho phép truy cập vào mục thứ i trong file index và trả về con trỏ tới và kích thước của mục đó. Lưu ý rằng phương thức này sử dụng functools.lru_cache để lưu trữ các kết quả truy cập trước đó trong bộ nhớ cache, giúp tăng tốc độ truy cập dữ liệu.'''
        return self._pointers[i], self._sizes[i]


# - - - - - - - - - - -


class MMapIndexedDataset(torch.utils.data.Dataset):
    '''Lớp MMapIndexedDataset trong Python là một lớp con của torch.utils.data.Dataset. Mục đích của lớp này là cung cấp một giao diện cung cấp dữ liệu từ một tệp nhị phân đã được chỉ mục bằng đối tượng Index.
    '''

    def __init__(self, path):
        '''Phương thức __init__ với đối số path là đường dẫn đến tệp nhị phân. Nó khởi tạo đối tượng Index với tệp *.idx tương ứng và tạo một bản đồ bộ nhớ của tệp nhị phân với phần mở rộng .bin. Bản đồ bộ nhớ được tạo với chế độ chỉ đọc và được sử dụng để lấy dữ liệu từ tệp nhị phân.
        '''
        super().__init__()
        self._index = Index(path + ".idx") 
        self._bin_buffer_mmap = np.memmap(path + ".bin", mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):  # Thu hồi bộ nhớ
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    def get(self, idx, offset=0, n=0):
        ptr, size = self._index[idx]
        if size <= n: return []
        if offset == "random": offset = np.random.randint(0, size - n)
        if n == 0: n = size - offset
        ptr += offset * np.dtype(self._index._dtype).itemsize
        return np.frombuffer(self._bin_buffer, dtype=self._index._dtype, count=n, offset=ptr)

    def get_global(self, offset=0, n=0):
        ptr, _ = self._index[0]
        ptr += offset * np.dtype(self._index._dtype).itemsize
        return np.frombuffer(self._bin_buffer, dtype=self._index._dtype, count=n, offset=ptr)

# - - - - - - - - - - -

import torch

class BinidxDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.fill_in_the_middle = ( args.fill_in_the_middle != "" )
        if ',' in args.data_file: # multiple data sources
            self.data = [ MMapIndexedDataset(x) for x in args.data_file.split(',') ]
            self.document_size = sum([ len(x) for x in self.data ])
            self.data_size = sum([ len(x._bin_buffer) for x in self.data ]) // 2
            sample_dix = self.data[0].get_global(offset=1000, n=5000).tolist()

        else: # single data source
            self.data = MMapIndexedDataset(args.data_file)
            self.document_size = len(self.data)
            self.data_size = len(self.data._bin_buffer) // 2
            sample_dix = self.data.get_global(offset=1000, n=5000).tolist()

        if args.epoch_steps_ratio > 0: # quên mất nó để làm gì rồi :D
            x = args.epoch_steps_ratio * args.epoch_steps
            self.samples_per_epoch = x * args.real_bsz

        if args.tokenizer == "symato":
            import sys; sys.path.append('..')
            from symato_16k import symato
            sample_data = symato.tids_to_utf8(sample_dix)
            self.vocab_size = 16384
        else:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor(model_file=args.vocab_file)
            sample_data = sp.decode(sample_dix)
            self.vocab_size = sp.vocab_size()

        print("\n- - - [ TRAIN DATA SAMPLE ] - - -\n\n", sample_data, "\n\n")
        print(f"Current vocab size = {self.vocab_size} (make sure it's correct)")
        print(f"Data has {self.data_size} tokens.\nData source(s) {args.data_file}")

        print(f"########## Bigdata stage {args.bigdata_stage} ##########")
        if args.bigdata_stage > 0:
            assert not isinstance(self.data, list), "TODO: hỗ trợ multi sources"
            assert args.tokens_per_hour > 1000 * 3600 # tốc độ học ít nhất phải 1k tokens / s
            self.samples_per_epoch = args.tokens_per_hour // args.ctx_len
            args.epoch_steps = self.samples_per_epoch // args.real_bsz
            # args.epoch_count là số lần lặp lại epoch_steps để lấy mẫu mỗi token 1 lần
            self.data_samples = (self.data_size - args.data_shift - 1) // args.ctx_len
            args.epoch_count = int(self.data_samples / self.samples_per_epoch / args.bigdata_portion)
            args.epoch_count += args.epoch_count_added


    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        req_len = self.args.ctx_len + 1  # cộng thêm một token là kết quả đầu ra

        if self.args.bigdata_stage > 0:
            offset = self.real_epoch * self.samples_per_epoch
            ii = offset + (idx * self.world_size) + self.global_rank            
            i = self.args.data_shift + ii * self.args.ctx_len
            if self.global_rank == 0 and idx % 50_000 == 0:
                print(f">>> Sampling: idx/ii {idx}/{ii}, i/n {i}/{self.data_size}")
            dix = self.data.get_global(offset=i, n=req_len).tolist()
        else:
            if isinstance(self.data, list): data = random.choice(self.data)
            else: data = self.data # random select data source
            data_size = len(data._bin_buffer) // 2
            document_size = len(data)
            # cheat: pick a random spot in dataset
            if np.random.uniform() > self.args.sampling_in_doc:
                n = 0; tids = None
                while (n < req_len):
                    doc_id = np.random.randint(0, document_size)
                    tids = data.get(doc_id, offset="random", n=req_len)
                    n = len(tids)
                dix = tids.tolist()
            else:
                i = np.random.randint(0, data_size - req_len)
                dix = data.get_global(offset=i, n=req_len).tolist()

        if self.fill_in_the_middle:
            if 500 > np.random.randint(0, 1000):
                sep = [3] # "\n" cho sentencepiece, 3 cho symato (16 là end-of-text cho symato)
                one_third = req_len // 3
                two_third = one_third * 2
                prefix = dix[0:one_third - 1] + sep
                middle = dix[one_third:two_third - 1] + sep
                suffix = dix[two_third:req_len]
                dix = suffix + prefix + middle

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
