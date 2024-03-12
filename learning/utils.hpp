/**
 * Useful data structures or functions
*/

#pragma once

struct DLinkNode {
    DLinkNode* prev;
    DLinkNode* next;
    short key;
    int value;

    DLinkNode(int key = -1, int val = -1): prev(nullptr), next(nullptr), key(key), value(val) {}
    DLinkNode(DLinkNode* p, DLinkNode* n, int key, int val = -1): prev(p), next(n), key(key), value(val) {}
};
