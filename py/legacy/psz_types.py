from ctypes import (
    c_int,
    c_char,
    c_bool,
    c_size_t,
    c_double,
    c_float,
    c_uint32,
    c_uint8,
    c_void_p,
    Structure,
    POINTER,
    Union,
)


class psz_dtype(c_int):
    """
    psz_dtype is a subclass of c_int that represents various data types.
    """

    F4 = 0
    F8 = 1
    U1 = 2
    U2 = 3
    U4 = 4
    U8 = 5
    I1 = 6
    I2 = 7
    I4 = 8
    I8 = 9
    ULL = 10


class psz_len3(Structure):
    _fields_ = [("x", c_size_t), ("y", c_size_t), ("z", c_size_t)]


class psz_predtype(c_int):
    Lorenzo = 0
    LorenzoZigZag = 1
    LorenzoProto = 2
    Spline = 3


class psz_histogramtype(c_int):
    HistogramGeneric = 0
    HistogramSparse = 1
    HistogramNull = 2


class psz_codectype(c_int):
    Huffman = 0
    HuffmanRevisit = 1
    FZGPUCodec = 2
    RunLength = 3


class psz_device(c_int):
    CPU = 0
    NVGPU = 1
    AMDGPU = 2
    INTELGPU = 3


class psz_mode(c_int):
    Abs = 0
    Rel = 1


class psz_error_status(c_int):
    _SUCCESS = 0
    _FAIL_GENERAL = 1
    _FAIL_UNSUPPORTED_DTYPE = 2
    _NOT_IMPLEMENTED = 3


# class psz_context(Structure):
#     _fields_ = [
#         ("device", psz_device),
#         ("dtype", psz_dtype),
#         ("pred_type", psz_predtype),
#         ("hist_type", psz_histogramtype),
#         ("codec1_type", psz_codectype),
#         ("mode", psz_mode),
#         ("eb", c_double),
#         ("dict_size", c_int),
#         ("radius", c_int),
#         ("prebuilt_bklen", c_int),
#         ("prebuilt_nbk", c_int),
#         ("nz_density", c_float),
#         ("nz_density_factor", c_float),
#         ("vle_sublen", c_int),
#         ("vle_pardeg", c_int),
#         ("x", c_uint32),
#         ("y", c_uint32),
#         ("z", c_uint32),
#         ("w", c_uint32),
#         ("data_len", c_size_t),
#         ("splen", c_size_t),
#         ("ndim", c_int),
#         ("last_error", psz_error_status),
#         ("user_input_eb", c_double),
#         ("logging_min", c_double),
#         ("logging_max", c_double),
#         ("demodata_name", c_char * 40),
#         ("opath", c_char * 200),
#         ("file_input", c_char * 500),
#         ("file_compare", c_char * 500),
#         ("file_prebuilt_hist_top1", c_char * 500),
#         ("file_prebuilt_hfbk", c_char * 500),
#         ("char_meta_eb", c_char * 16),
#         ("char_predictor_name", c_char * len("lorenzo-zigzag")),
#         ("char_hist_name", c_char * len("histogram-centrality")),
#         ("char_codec1_name", c_char * len("huffman-revisit")),
#         ("char_codec2_name", c_char * len("huffman-revisit")),
#         ("dump_quantcode", c_bool),
#         ("dump_hist", c_bool),
#         ("dump_full_hf", c_bool),
#         ("task_construct", c_bool),
#         ("task_reconstruct", c_bool),
#         ("task_dryrun", c_bool),
#         ("task_experiment", c_bool),
#         ("prep_binning", c_bool),
#         ("prep_prescan", c_bool),
#         ("use_demodata", c_bool),
#         ("use_autotune_phf", c_bool),
#         ("use_gpu_verify", c_bool),
#         ("use_prebuilt_hfbk", c_bool),
#         ("skip_tofile", c_bool),
#         ("skip_hf", c_bool),
#         ("report_time", c_bool),
#         ("report_cr", c_bool),
#         ("verbose", c_bool),
#         ("there_is_memerr", c_bool),
#     ]


class psz_header(Structure):
    PSZHEADER_END = 4
    _fields_ = [
        ("dtype", psz_dtype),
        ("pred_type", psz_predtype),
        ("hist_type", psz_histogramtype),
        ("codec1_type", psz_codectype),
        ("entry", c_uint32 * (PSZHEADER_END + 1)),
        ("splen", c_int),
        ("x", c_uint32),
        ("y", c_uint32),
        ("z", c_uint32),
        ("w", c_uint32),
        ("user_input_eb", c_double),
        ("eb", c_double),
        ("radius", c_uint32, 16),
        ("vle_pardeg", c_uint32),
        ("logging_min", c_double),
        ("logging_max", c_double),
        ("logging_mode", psz_mode),
    ]
