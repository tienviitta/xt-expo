#ifndef ENCDL_H_
#define ENCDL_H_

typedef struct params_s {
    int A;
    int P;
    int K;
    int E;
    int N;
} params_s;

void encDl(params_s *params);

#endif // ENCDL_H_