# parallelism



# API (draft)

Structure

\begin{multicols}{2}
\begin{verbatim}
cuSZ driver program
│
├── DryRun
│   ├── c_lorenzo_1d1l
│   ├── c_lorenzo_2d1l
│   └── c_lorenzo_3d1l
│
├── Dual-Quant
│   ├── c_lorenzo_1d1l
│   ├── c_lorenzo_2d1l
│   └── c_lorenzo_3d1l
│
├── reversed Dual-Quant
│   ├── x_lorenzo_1d1l
│   ├── x_lorenzo_2d1l
│   └── x_lorenzo_3d1l
...
\end{verbatim}

\vfill\null
\columnbreak

\begin{verbatim}
...
│
├── HuffmanEncode
│   ├── 0.GetFrequency
│   ├── 1.InitHuffTreeAndGetCodebook
│   ├── 2.GetCanonicalCode
│   ├── 3.EncodeFixedLen
│   └── 4.Deflate
│
├── HuffmanDecode
│   └── Decode
│       └── InflateChunkwise
│
└── verification

\end{verbatim}
\end{multicols}

## `Dual-Quant` and reversed `Dual-Quant`

## histogramming

## Huffman codec
### 