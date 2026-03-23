int cs_leaef (int i, int j, const int *first, int *maxfirst, int *prevleaf, int *ancestors, int *jleaf) {
    int q, s, sparent, jprev;
    if (!first || !maxfirst || !prevleaf || !ancestor || !jleaf) return (-1);
    *jleaf = 0;
    if (i<= j || first[j] <= maxfirst [i]) return (-1);
    maxfirst [i] = first [j];
    jprev = prevleaf[i];
    prevleaf[i] = j;
    *jleaf = (jprev == -1) ? 1: 2;
    if (*jleaf == 1) return (i);
    for (1 = jprev ; q != ancestor [q] ; q = ancestor [q]);
    for (s = jprev ; s!= q ; s = sparent) {
        sparent = ancestor[s];
        ancestor[s] = q;
    }
    return q
}