/**
 *  @file Huffman.c
 *  @author Sheng Di
 *  @date Aug., 2016
 *  @brief Customized Huffman Encoding, Compression and Decompression functions
 *  (C) 2016 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef _sz_huffman_hh
#define _sz_huffman_hh

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace std;

namespace DesignVerification {

struct node_t {
    struct node_t *left, *right;
    size_t         freq;
    char           t;  // in_node:0; otherwise:1
    uint32_t       c;
};
typedef struct node_t* node;
typedef struct node_t  treenode_t;

typedef struct HuffmanTree {
    uint32_t       stateNum;
    uint32_t       allNodes;
    struct node_t* pool;
    node *         qqq, *qq;  // the root node of the HuffmanTree is qq[1]
    int            n_nodes;   // n_nodes is for compression
    int            qend;
    uint64_t**     code;
    uint8_t*       cout;
    int            n_inode;  // n_inode is for decompression
} HuffmanTree;

HuffmanTree* createHuffmanTree(int stateNum);
HuffmanTree* createDefaultHuffmanTree();

node     new_node(HuffmanTree* huffmanTree, size_t freq, uint32_t c, node a, node b);
node     new_node2(HuffmanTree* huffmanTree, uint32_t c, uint8_t t);
void     qinsert(HuffmanTree* huffmanTree, node n);
node     qremove(HuffmanTree* huffmanTree);
void     build_code(HuffmanTree* huffmanTree, node n, int len, uint64_t out1, uint64_t out2);
void     initHuffman(HuffmanTree* huffmanTree, const int* s, size_t length);
void     HuffmanEncode(HuffmanTree* huffmanTree, const int* s, size_t length, uint8_t* out, size_t* outSize);
void     decode(const uint8_t* s, size_t targetLength, node t, int* out);
void     pad_tree_uchar(HuffmanTree* huffmanTree, uint8_t* L, uint8_t* R, uint32_t* C, uint8_t* t, uint32_t i, node root);
void     pad_tree_ushort(HuffmanTree* huffmanTree, uint16_t* L, uint16_t* R, uint32_t* C, uint8_t* t, uint32_t i, node root);
void     pad_tree_uint(HuffmanTree* huffmanTree, uint32_t* L, uint32_t* R, uint32_t* C, uint8_t* t, uint32_t i, node root);
uint32_t convert_HuffTree_to_bytes_anyStates(HuffmanTree* huffmanTree, int nodeCount, uint8_t** out);
void     unpad_tree_uchar(HuffmanTree* huffmanTree, uint8_t* L, uint8_t* R, uint32_t* C, uint8_t* t, uint32_t i, node root);
void     unpad_tree_ushort(HuffmanTree* huffmanTree, uint16_t* L, uint16_t* R, uint32_t* C, uint8_t* t, uint32_t i, node root);
void     unpad_tree_uint(HuffmanTree* huffmanTree, uint32_t* L, uint32_t* R, uint32_t* C, uint8_t* t, uint32_t i, node root);
node     reconstruct_HuffTree_from_bytes_anyStates(HuffmanTree* huffmanTree, const uint8_t* bytes, int nodeCount);

void encode_withTree(HuffmanTree* huffmanTree, const int* s, size_t length, uint8_t** out, size_t* outSize);
void decode_withTree(HuffmanTree* huffmanTree, const uint8_t* s, size_t targetLength, int* out);

void SZ_ReleaseHuffman(HuffmanTree* huffmanTree);

// copied from sz, adding const attribute
// to be modifiled later
// also copied some auxiliary functions
static int  CPU_sysEndianType = 0;
inline void longToBytes_bigEndian(uint8_t* b, uint64_t num) {
    b[0] = (uint8_t)(num >> 56);
    b[1] = (uint8_t)(num >> 48);
    b[2] = (uint8_t)(num >> 40);
    b[3] = (uint8_t)(num >> 32);
    b[4] = (uint8_t)(num >> 24);
    b[5] = (uint8_t)(num >> 16);
    b[6] = (uint8_t)(num >> 8);
    b[7] = (uint8_t)(num);
}

inline void intToBytes_bigEndian(uint8_t* b, uint32_t num) {
    b[0] = (uint8_t)(num >> 24);
    b[1] = (uint8_t)(num >> 16);
    b[2] = (uint8_t)(num >> 8);
    b[3] = (uint8_t)(num);
}

inline int bytesToInt_bigEndian(const uint8_t* bytes) {
    int temp = 0;
    int res  = 0;

    res <<= 8;
    temp = bytes[0] & 0xff;
    res |= temp;

    res <<= 8;
    temp = bytes[1] & 0xff;
    res |= temp;

    res <<= 8;
    temp = bytes[2] & 0xff;
    res |= temp;

    res <<= 8;
    temp = bytes[3] & 0xff;
    res |= temp;

    return res;
}

// auxiliary functions done

HuffmanTree* createHuffmanTree(int stateNum) {
    HuffmanTree* huffmanTree = (HuffmanTree*)malloc(sizeof(HuffmanTree));
    memset(huffmanTree, 0, sizeof(HuffmanTree));
    huffmanTree->stateNum = stateNum;
    huffmanTree->allNodes = 2 * stateNum;

    huffmanTree->pool = (struct node_t*)malloc(huffmanTree->allNodes * 2 * sizeof(struct node_t));
    huffmanTree->qqq  = (node*)malloc(huffmanTree->allNodes * 2 * sizeof(node));
    huffmanTree->code = (uint64_t**)malloc(huffmanTree->stateNum * sizeof(uint64_t*));
    huffmanTree->cout = (uint8_t*)malloc(huffmanTree->stateNum * sizeof(uint8_t));

    memset(huffmanTree->pool, 0, huffmanTree->allNodes * 2 * sizeof(struct node_t));
    memset(huffmanTree->qqq, 0, huffmanTree->allNodes * 2 * sizeof(node));
    memset(huffmanTree->code, 0, huffmanTree->stateNum * sizeof(uint64_t*));
    memset(huffmanTree->cout, 0, huffmanTree->stateNum * sizeof(uint8_t));
    huffmanTree->qq      = huffmanTree->qqq - 1;
    huffmanTree->n_nodes = 0;
    huffmanTree->n_inode = 0;
    huffmanTree->qend    = 1;

    return huffmanTree;
}

HuffmanTree* createDefaultHuffmanTree() {
    int maxRangeRadius = 32768;
    int stateNum       = maxRangeRadius << 1;  //*2

    return createHuffmanTree(stateNum);
}

node new_node(HuffmanTree* huffmanTree, size_t freq, uint32_t c, node a, node b) {
    node n = huffmanTree->pool + huffmanTree->n_nodes++;
    if (freq) {
        n->c    = c;
        n->freq = freq;
        n->t    = 1;
    } else {
        n->left  = a;
        n->right = b;
        n->freq  = a->freq + b->freq;
        n->t     = 0;
        // n->c = 0;
    }
    return n;
}

node new_node2(HuffmanTree* huffmanTree, uint32_t c, uint8_t t) {
    huffmanTree->pool[huffmanTree->n_nodes].c = c;
    huffmanTree->pool[huffmanTree->n_nodes].t = t;
    return huffmanTree->pool + huffmanTree->n_nodes++;
}

/* priority queue */
void qinsert(HuffmanTree* huffmanTree, node n) {
    int j, i = huffmanTree->qend++;
    while ((j = (i >> 1)))  // j=i/2
    {
        if (huffmanTree->qq[j]->freq <= n->freq) break;
        huffmanTree->qq[i] = huffmanTree->qq[j], i = j;
    }
    huffmanTree->qq[i] = n;
}

node qremove(HuffmanTree* huffmanTree) {
    int  i, l;
    node n = huffmanTree->qq[i = 1];

    if (huffmanTree->qend < 2) return 0;
    huffmanTree->qend--;
    while ((l = (i << 1)) < huffmanTree->qend)  // l=(i*2)
    {
        if (l + 1 < huffmanTree->qend && huffmanTree->qq[l + 1]->freq < huffmanTree->qq[l]->freq) l++;
        huffmanTree->qq[i] = huffmanTree->qq[l], i = l;
    }
    huffmanTree->qq[i] = huffmanTree->qq[huffmanTree->qend];
    return n;
}

/* walk the tree and put 0s and 1s */
/**
 * @out1 should be set to 0.
 * @out2 should be 0 as well.
 * @index: the index of the byte
 * */
void build_code(HuffmanTree* huffmanTree, node n, int len, uint64_t out1, uint64_t out2) {
    if (n->t) {
        huffmanTree->code[n->c] = (uint64_t*)malloc(2 * sizeof(uint64_t));
        if (len <= 64) {
            (huffmanTree->code[n->c])[0] = out1 << (64 - len);
            (huffmanTree->code[n->c])[1] = out2;
        } else {
            (huffmanTree->code[n->c])[0] = out1;
            (huffmanTree->code[n->c])[1] = out2 << (128 - len);
        }
        huffmanTree->cout[n->c] = (uint8_t)len;
        return;
    }
    int index = len >> 6;  //=len/64
    if (index == 0) {
        out1 = out1 << 1;
        out1 = out1 | 0;
        build_code(huffmanTree, n->left, len + 1, out1, 0);
        out1 = out1 | 1;
        build_code(huffmanTree, n->right, len + 1, out1, 0);
    } else {
        if (len % 64 != 0) out2 = out2 << 1;
        out2 = out2 | 0;
        build_code(huffmanTree, n->left, len + 1, out1, out2);
        out2 = out2 | 1;
        build_code(huffmanTree, n->right, len + 1, out1, out2);
    }
}

/**
 * Compute the frequency of the data and build the Huffman tree
 * @param HuffmanTree* huffmanTree (output)
 * @param int *s (input)
 * @param size_t length (input)
 * */
void initHuffman(HuffmanTree* huffmanTree, const int* s, size_t length) {
    size_t  i, index;
    size_t* freq = (size_t*)malloc(huffmanTree->allNodes * sizeof(size_t));
    memset(freq, 0, huffmanTree->allNodes * sizeof(size_t));
    for (i = 0; i < length; i++) {
        index = s[i];
        freq[index]++;
    }
    /*
        for (i = 0; i < huffmanTree->allNodes; i++)
            if (freq[i]) qinsert(huffmanTree, new_node(huffmanTree, freq[i], i, 0, 0));
    */
    size_t minimal = SIZE_MAX;
    for (i = 0; i < huffmanTree->allNodes; i++) {
        if (freq[i]) {
            minimal = minimal > freq[i] ? freq[i] : minimal;
            qinsert(huffmanTree, new_node(huffmanTree, freq[i], i, 0, 0));
        }
    }
    //    cout << "minimal freq\t" << minimal << endl;
    //    cout << "minimal freq%\t" << static_cast<double>(minimal) / length << endl;
    //    cout << "minimal freq reciprocal\t" << length / static_cast<double>(minimal) << endl;

    while (huffmanTree->qend > 2) qinsert(huffmanTree, new_node(huffmanTree, 0, 0, qremove(huffmanTree), qremove(huffmanTree)));

    build_code(huffmanTree, huffmanTree->qq[1], 0, 0, 0);
    free(freq);
}

void HuffmanEncode(HuffmanTree* huffmanTree, const int* s, size_t length, uint8_t* out, size_t* outSize) {
    size_t   i       = 0;
    uint8_t  bitSize = 0, byteSize, byteSizep;
    int      state;
    uint8_t* p        = out;
    int      lackBits = 0;
    // long totalBitSize = 0, maxBitSize = 0, bitSize21 = 0, bitSize32 = 0;
    for (i = 0; i < length; i++) {
        state   = s[i];
        bitSize = huffmanTree->cout[state];

        if (lackBits == 0) {
            byteSize  = bitSize % 8 == 0 ? bitSize / 8 : bitSize / 8 + 1;  // it's equal to the number of bytes involved (for *outSize)
            byteSizep = bitSize / 8;                                       // it's used to move the pointer p for next data
            if (byteSize <= 8) {
                longToBytes_bigEndian(p, (huffmanTree->code[state])[0]);
                p += byteSizep;
            } else  // byteSize>8
            {
                longToBytes_bigEndian(p, (huffmanTree->code[state])[0]);
                p += 8;
                longToBytes_bigEndian(p, (huffmanTree->code[state])[1]);
                p += (byteSizep - 8);
            }
            *outSize += byteSize;
            lackBits = bitSize % 8 == 0 ? 0 : 8 - bitSize % 8;
        } else {
            *p = (*p) | (uint8_t)((huffmanTree->code[state])[0] >> (64 - lackBits));
            if (lackBits < bitSize) {
                p++;
                //(*outSize)++;
                long newCode = (huffmanTree->code[state])[0] << lackBits;
                longToBytes_bigEndian(p, newCode);

                if (bitSize <= 64) {
                    bitSize -= lackBits;
                    byteSize  = bitSize % 8 == 0 ? bitSize / 8 : bitSize / 8 + 1;
                    byteSizep = bitSize / 8;
                    p += byteSizep;
                    (*outSize) += byteSize;
                    lackBits = bitSize % 8 == 0 ? 0 : 8 - bitSize % 8;
                } else  // bitSize > 64
                {
                    byteSizep = 7;  // must be 7 bytes, because lackBits!=0
                    p += byteSizep;
                    (*outSize) += byteSize;

                    bitSize -= 64;
                    if (lackBits < bitSize) {
                        *p = (*p) | (uint8_t)((huffmanTree->code[state])[0] >> (64 - lackBits));
                        p++;
                        //(*outSize)++;
                        newCode = (huffmanTree->code[state])[1] << lackBits;
                        longToBytes_bigEndian(p, newCode);
                        bitSize -= lackBits;
                        byteSize  = bitSize % 8 == 0 ? bitSize / 8 : bitSize / 8 + 1;
                        byteSizep = bitSize / 8;
                        p += byteSizep;
                        (*outSize) += byteSize;
                        lackBits = bitSize % 8 == 0 ? 0 : 8 - bitSize % 8;
                    } else  // lackBits >= bitSize
                    {
                        *p = (*p) | (uint8_t)((huffmanTree->code[state])[0] >> (64 - bitSize));
                        lackBits -= bitSize;
                    }
                }
            } else  // lackBits >= bitSize
            {
                lackBits -= bitSize;
                if (lackBits == 0) p++;
            }
        }
    }
}

void decode(const uint8_t* s, size_t targetLength, node t, int* out) {
    size_t i = 0, byteIndex = 0, count = 0;
    int    r;
    node   n = t;

    if (n->t)  // root->t==1 means that all state values are the same (constant)
    {
        for (count = 0; count < targetLength; count++) out[count] = n->c;
        return;
    }

    for (i = 0; count < targetLength; i++) {
        byteIndex = i >> 3;  // i/8
        r         = i % 8;
        if (((s[byteIndex] >> (7 - r)) & 0x01) == 0)
            n = n->left;
        else
            n = n->right;

        if (n->t) {
            // putchar(n->c);
            out[count] = n->c;
            n          = t;
            count++;
        }
    }
    //	putchar('\n');
    if (t != n) printf("garbage input\n");
    return;
}

void pad_tree_uchar(HuffmanTree* huffmanTree, uint8_t* L, uint8_t* R, uint32_t* C, uint8_t* t, uint32_t i, node root) {
    C[i]       = root->c;
    t[i]       = root->t;
    node lroot = root->left;
    if (lroot != 0) {
        huffmanTree->n_inode++;
        L[i] = huffmanTree->n_inode;
        pad_tree_uchar(huffmanTree, L, R, C, t, huffmanTree->n_inode, lroot);
    }
    node rroot = root->right;
    if (rroot != 0) {
        huffmanTree->n_inode++;
        R[i] = huffmanTree->n_inode;
        pad_tree_uchar(huffmanTree, L, R, C, t, huffmanTree->n_inode, rroot);
    }
}

void pad_tree_ushort(HuffmanTree* huffmanTree, uint16_t* L, uint16_t* R, uint32_t* C, uint8_t* t, uint32_t i, node root) {
    C[i]       = root->c;
    t[i]       = root->t;
    node lroot = root->left;
    if (lroot != 0) {
        huffmanTree->n_inode++;
        L[i] = huffmanTree->n_inode;
        pad_tree_ushort(huffmanTree, L, R, C, t, huffmanTree->n_inode, lroot);
    }
    node rroot = root->right;
    if (rroot != 0) {
        huffmanTree->n_inode++;
        R[i] = huffmanTree->n_inode;
        pad_tree_ushort(huffmanTree, L, R, C, t, huffmanTree->n_inode, rroot);
    }
}

void pad_tree_uint(HuffmanTree* huffmanTree, uint32_t* L, uint32_t* R, uint32_t* C, uint8_t* t, uint32_t i, node root) {
    C[i]       = root->c;
    t[i]       = root->t;
    node lroot = root->left;
    if (lroot != 0) {
        huffmanTree->n_inode++;
        L[i] = huffmanTree->n_inode;
        pad_tree_uint(huffmanTree, L, R, C, t, huffmanTree->n_inode, lroot);
    }
    node rroot = root->right;
    if (rroot != 0) {
        huffmanTree->n_inode++;
        R[i] = huffmanTree->n_inode;
        pad_tree_uint(huffmanTree, L, R, C, t, huffmanTree->n_inode, rroot);
    }
}

uint32_t convert_HuffTree_to_bytes_anyStates(HuffmanTree* huffmanTree, int nodeCount, uint8_t** out) {
    if (nodeCount <= 256) {
        uint8_t* L = (uint8_t*)malloc(nodeCount * sizeof(uint8_t));
        memset(L, 0, nodeCount * sizeof(uint8_t));
        uint8_t* R = (uint8_t*)malloc(nodeCount * sizeof(uint8_t));
        memset(R, 0, nodeCount * sizeof(uint8_t));
        uint32_t* C = (uint32_t*)malloc(nodeCount * sizeof(uint32_t));
        memset(C, 0, nodeCount * sizeof(uint32_t));
        uint8_t* t = (uint8_t*)malloc(nodeCount * sizeof(uint8_t));
        memset(t, 0, nodeCount * sizeof(uint8_t));

        pad_tree_uchar(huffmanTree, L, R, C, t, 0, huffmanTree->qq[1]);

        uint32_t totalSize = 1 + 3 * nodeCount * sizeof(uint8_t) + nodeCount * sizeof(uint32_t);
        *out               = (uint8_t*)malloc(totalSize * sizeof(uint8_t));
        (*out)[0]          = (uint8_t)CPU_sysEndianType;
        memcpy(*out + 1, L, nodeCount * sizeof(uint8_t));
        memcpy((*out) + 1 + nodeCount * sizeof(uint8_t), R, nodeCount * sizeof(uint8_t));
        memcpy((*out) + 1 + 2 * nodeCount * sizeof(uint8_t), C, nodeCount * sizeof(uint32_t));
        memcpy((*out) + 1 + 2 * nodeCount * sizeof(uint8_t) + nodeCount * sizeof(uint32_t), t, nodeCount * sizeof(uint8_t));
        free(L);
        free(R);
        free(C);
        free(t);
        return totalSize;

    } else if (nodeCount <= 65536) {
        uint16_t* L = (uint16_t*)malloc(nodeCount * sizeof(uint16_t));
        memset(L, 0, nodeCount * sizeof(uint16_t));
        uint16_t* R = (uint16_t*)malloc(nodeCount * sizeof(uint16_t));
        memset(R, 0, nodeCount * sizeof(uint16_t));
        uint32_t* C = (uint32_t*)malloc(nodeCount * sizeof(uint32_t));
        memset(C, 0, nodeCount * sizeof(uint32_t));
        uint8_t* t = (uint8_t*)malloc(nodeCount * sizeof(uint8_t));
        memset(t, 0, nodeCount * sizeof(uint8_t));
        pad_tree_ushort(huffmanTree, L, R, C, t, 0, huffmanTree->qq[1]);
        uint32_t totalSize = 1 + 2 * nodeCount * sizeof(uint16_t) + nodeCount * sizeof(uint8_t) + nodeCount * sizeof(uint32_t);
        *out               = (uint8_t*)malloc(totalSize);
        (*out)[0]          = (uint8_t)CPU_sysEndianType;
        memcpy(*out + 1, L, nodeCount * sizeof(uint16_t));
        memcpy((*out) + 1 + nodeCount * sizeof(uint16_t), R, nodeCount * sizeof(uint16_t));
        memcpy((*out) + 1 + 2 * nodeCount * sizeof(uint16_t), C, nodeCount * sizeof(uint32_t));
        memcpy((*out) + 1 + 2 * nodeCount * sizeof(uint16_t) + nodeCount * sizeof(uint32_t), t, nodeCount * sizeof(uint8_t));
        free(L);
        free(R);
        free(C);
        free(t);
        return totalSize;
    } else  // nodeCount>65536
    {
        uint32_t* L = (uint32_t*)malloc(nodeCount * sizeof(uint32_t));
        memset(L, 0, nodeCount * sizeof(uint32_t));
        uint32_t* R = (uint32_t*)malloc(nodeCount * sizeof(uint32_t));
        memset(R, 0, nodeCount * sizeof(uint32_t));
        uint32_t* C = (uint32_t*)malloc(nodeCount * sizeof(uint32_t));
        memset(C, 0, nodeCount * sizeof(uint32_t));
        uint8_t* t = (uint8_t*)malloc(nodeCount * sizeof(uint8_t));
        memset(t, 0, nodeCount * sizeof(uint8_t));
        pad_tree_uint(huffmanTree, L, R, C, t, 0, huffmanTree->qq[1]);

        // debug
        // node root = new_node2(0,0);
        // unpad_tree_uint(L,R,C,t,0,root);

        uint32_t totalSize = 1 + 3 * nodeCount * sizeof(uint32_t) + nodeCount * sizeof(uint8_t);
        *out               = (uint8_t*)malloc(totalSize);
        (*out)[0]          = (uint8_t)CPU_sysEndianType;
        memcpy(*out + 1, L, nodeCount * sizeof(uint32_t));
        memcpy((*out) + 1 + nodeCount * sizeof(uint32_t), R, nodeCount * sizeof(uint32_t));
        memcpy((*out) + 1 + 2 * nodeCount * sizeof(uint32_t), C, nodeCount * sizeof(uint32_t));
        memcpy((*out) + 1 + 3 * nodeCount * sizeof(uint32_t), t, nodeCount * sizeof(uint8_t));
        free(L);
        free(R);
        free(C);
        free(t);
        return totalSize;
    }
}

void unpad_tree_uchar(HuffmanTree* huffmanTree, uint8_t* L, uint8_t* R, uint32_t* C, uint8_t* t, uint32_t i, node root) {
    // root->c = C[i];
    if (root->t == 0) {
        uint8_t l, r;
        l = L[i];
        if (l != 0) {
            node lroot = new_node2(huffmanTree, C[l], t[l]);
            root->left = lroot;
            unpad_tree_uchar(huffmanTree, L, R, C, t, l, lroot);
        }
        r = R[i];
        if (r != 0) {
            node rroot  = new_node2(huffmanTree, C[r], t[r]);
            root->right = rroot;
            unpad_tree_uchar(huffmanTree, L, R, C, t, r, rroot);
        }
    }
}

void unpad_tree_ushort(HuffmanTree* huffmanTree, uint16_t* L, uint16_t* R, uint32_t* C, uint8_t* t, uint32_t i, node root) {
    // root->c = C[i];
    if (root->t == 0) {
        uint16_t l, r;
        l = L[i];
        if (l != 0) {
            node lroot = new_node2(huffmanTree, C[l], t[l]);
            root->left = lroot;
            unpad_tree_ushort(huffmanTree, L, R, C, t, l, lroot);
        }
        r = R[i];
        if (r != 0) {
            node rroot  = new_node2(huffmanTree, C[r], t[r]);
            root->right = rroot;
            unpad_tree_ushort(huffmanTree, L, R, C, t, r, rroot);
        }
    }
}

void unpad_tree_uint(HuffmanTree* huffmanTree, uint32_t* L, uint32_t* R, uint32_t* C, uint8_t* t, uint32_t i, node root) {
    // root->c = C[i];
    if (root->t == 0) {
        uint32_t l, r;
        l = L[i];
        if (l != 0) {
            node lroot = new_node2(huffmanTree, C[l], t[l]);
            root->left = lroot;
            unpad_tree_uint(huffmanTree, L, R, C, t, l, lroot);
        }
        r = R[i];
        if (r != 0) {
            node rroot  = new_node2(huffmanTree, C[r], t[r]);
            root->right = rroot;
            unpad_tree_uint(huffmanTree, L, R, C, t, r, rroot);
        }
    }
}

node reconstruct_HuffTree_from_bytes_anyStates(HuffmanTree* huffmanTree, const uint8_t* bytes, int nodeCount) {
    if (nodeCount <= 256) {
        uint8_t* L = (uint8_t*)malloc(nodeCount * sizeof(uint8_t));
        memset(L, 0, nodeCount * sizeof(uint8_t));
        uint8_t* R = (uint8_t*)malloc(nodeCount * sizeof(uint8_t));
        memset(R, 0, nodeCount * sizeof(uint8_t));
        uint32_t* C = (uint32_t*)malloc(nodeCount * sizeof(uint32_t));
        memset(C, 0, nodeCount * sizeof(uint32_t));
        uint8_t* t = (uint8_t*)malloc(nodeCount * sizeof(uint8_t));
        memset(t, 0, nodeCount * sizeof(uint8_t));
        // uint8_t cmpCPU_sysEndianType = bytes[0];

        memcpy(L, bytes + 1, nodeCount * sizeof(uint8_t));
        memcpy(R, bytes + 1 + nodeCount * sizeof(uint8_t), nodeCount * sizeof(uint8_t));
        memcpy(C, bytes + 1 + 2 * nodeCount * sizeof(uint8_t), nodeCount * sizeof(uint32_t));
        memcpy(t, bytes + 1 + 2 * nodeCount * sizeof(uint8_t) + nodeCount * sizeof(uint32_t), nodeCount * sizeof(uint8_t));
        node root = new_node2(huffmanTree, C[0], t[0]);
        unpad_tree_uchar(huffmanTree, L, R, C, t, 0, root);
        free(L);
        free(R);
        free(C);
        free(t);
        return root;
    } else if (nodeCount <= 65536) {
        uint16_t* L = (uint16_t*)malloc(nodeCount * sizeof(uint16_t));
        memset(L, 0, nodeCount * sizeof(uint16_t));
        uint16_t* R = (uint16_t*)malloc(nodeCount * sizeof(uint16_t));
        memset(R, 0, nodeCount * sizeof(uint16_t));
        uint32_t* C = (uint32_t*)malloc(nodeCount * sizeof(uint32_t));
        memset(C, 0, nodeCount * sizeof(uint32_t));
        uint8_t* t = (uint8_t*)malloc(nodeCount * sizeof(uint8_t));
        memset(t, 0, nodeCount * sizeof(uint8_t));

        // uint8_t cmpCPU_sysEndianType = bytes[0];

        memcpy(L, bytes + 1, nodeCount * sizeof(uint16_t));
        memcpy(R, bytes + 1 + nodeCount * sizeof(uint16_t), nodeCount * sizeof(uint16_t));
        memcpy(C, bytes + 1 + 2 * nodeCount * sizeof(uint16_t), nodeCount * sizeof(uint32_t));

        memcpy(t, bytes + 1 + 2 * nodeCount * sizeof(uint16_t) + nodeCount * sizeof(uint32_t), nodeCount * sizeof(uint8_t));

        node root = new_node2(huffmanTree, 0, 0);
        unpad_tree_ushort(huffmanTree, L, R, C, t, 0, root);
        free(L);
        free(R);
        free(C);
        free(t);
        return root;
    } else  // nodeCount>65536
    {
        uint32_t* L = (uint32_t*)malloc(nodeCount * sizeof(uint32_t));
        memset(L, 0, nodeCount * sizeof(uint32_t));
        uint32_t* R = (uint32_t*)malloc(nodeCount * sizeof(uint32_t));
        memset(R, 0, nodeCount * sizeof(uint32_t));
        uint32_t* C = (uint32_t*)malloc(nodeCount * sizeof(uint32_t));
        memset(C, 0, nodeCount * sizeof(uint32_t));
        uint8_t* t = (uint8_t*)malloc(nodeCount * sizeof(uint8_t));
        memset(t, 0, nodeCount * sizeof(uint8_t));
        // uint8_t cmpCPU_sysEndianType = bytes[0];

        memcpy(L, bytes + 1, nodeCount * sizeof(uint32_t));
        memcpy(R, bytes + 1 + nodeCount * sizeof(uint32_t), nodeCount * sizeof(uint32_t));
        memcpy(C, bytes + 1 + 2 * nodeCount * sizeof(uint32_t), nodeCount * sizeof(uint32_t));

        memcpy(t, bytes + 1 + 3 * nodeCount * sizeof(uint32_t), nodeCount * sizeof(uint8_t));

        node root = new_node2(huffmanTree, 0, 0);
        unpad_tree_uint(huffmanTree, L, R, C, t, 0, root);
        free(L);
        free(R);
        free(C);
        free(t);
        return root;
    }
}

void encode_withTree(HuffmanTree* huffmanTree, const int* s, size_t length, uint8_t** out, size_t* outSize) {
    size_t   i;
    int      nodeCount = 0;
    uint8_t *treeBytes, buffer[4];

    initHuffman(huffmanTree, s, length);
    for (i = 0; i < huffmanTree->stateNum; i++)
        if (huffmanTree->code[i]) nodeCount++;
    nodeCount             = nodeCount * 2 - 1;
    uint32_t treeByteSize = convert_HuffTree_to_bytes_anyStates(huffmanTree, nodeCount, &treeBytes);

    *out = (uint8_t*)malloc(length * sizeof(int) + treeByteSize);
    intToBytes_bigEndian(buffer, nodeCount);
    memcpy(*out, buffer, 4);
    intToBytes_bigEndian(buffer, huffmanTree->stateNum / 2);  // real number of intervals
    memcpy(*out + 4, buffer, 4);
    memcpy(*out + 8, treeBytes, treeByteSize);
    free(treeBytes);
    size_t enCodeSize = 0;
    HuffmanEncode(huffmanTree, s, length, *out + 8 + treeByteSize, &enCodeSize);
    *outSize = 8 + treeByteSize + enCodeSize;

    // uint16_t state[length];
    // decode(*out+4+treeByteSize, enCodeSize, qqq[0], state);
    // printf("dataSeriesLength=%d",length );
}

/**
 * @par *out rememmber to allocate targetLength short_type data for it beforehand.
 *
 * */
void decode_withTree(HuffmanTree* huffmanTree, const uint8_t* s, size_t targetLength, int* out) {
    size_t encodeStartIndex;
    size_t nodeCount = bytesToInt_bigEndian(s);
    node   root      = reconstruct_HuffTree_from_bytes_anyStates(huffmanTree, s + 8, nodeCount);

    if (nodeCount <= 256)
        encodeStartIndex = 1 + 3 * nodeCount * sizeof(uint8_t) + nodeCount * sizeof(uint32_t);
    else if (nodeCount <= 65536)
        encodeStartIndex = 1 + 2 * nodeCount * sizeof(uint16_t) + nodeCount * sizeof(uint8_t) + nodeCount * sizeof(uint32_t);
    else
        encodeStartIndex = 1 + 3 * nodeCount * sizeof(uint32_t) + nodeCount * sizeof(uint8_t);
    decode(s + 8 + encodeStartIndex, targetLength, root, out);
}

void SZ_ReleaseHuffman(HuffmanTree* huffmanTree) {
    size_t i;
    free(huffmanTree->pool);
    huffmanTree->pool = nullptr;
    free(huffmanTree->qqq);
    huffmanTree->qqq = nullptr;
    for (i = 0; i < huffmanTree->stateNum; i++) {
        if (huffmanTree->code[i] != nullptr) free(huffmanTree->code[i]);
    }
    free(huffmanTree->code);
    huffmanTree->code = nullptr;
    free(huffmanTree->cout);
    huffmanTree->cout = nullptr;
    free(huffmanTree);
    huffmanTree = nullptr;
}
}  // namespace DesignVerification
#endif
