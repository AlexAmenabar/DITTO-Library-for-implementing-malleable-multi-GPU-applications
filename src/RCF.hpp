#ifndef RCF_HPP
#define RCF_HPP

typedef struct schInfo_t schInfo_t;

// policies
void utilization(schInfo_t *schInfo);
void topology(schInfo_t *schInfo);

// interface
void reconf(schInfo_t *schInfo);

#endif