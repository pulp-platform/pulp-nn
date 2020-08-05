#include "rt/rt_api.h"

unsigned int __rt_platform = 3;

rt_dev_t __rt_devices[] = {
  {"jtag_proxy", -1, 0, (void *)NULL, {{}}},
  {"uart", -1, 0, (void *)NULL, {{}}},
  {"jtag_proxy", -1, 0, (void *)NULL, {{}}},
  {"camera", -1, 0, (void *)NULL, {{}}},
  {"camera", -1, 0, (void *)&himax_desc, {{}}},
};

int __rt_nb_devices = 5;
