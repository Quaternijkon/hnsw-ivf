# Faiss 性能对比实验报告 (最终版)

## 实验目的
通过对比普通磁盘和内存磁盘的性能表现，分析磁盘I/O是否在高线程数下成为性能瓶颈。

## 测试环境
- 测试时间: Thu Oct  9 09:11:19 AM UTC 2025
- 系统: Linux 5.15.0-97-generic
- 内存: 219Gi
- 磁盘信息: tmpfs            22G  3.1M   22G   1% /run
/dev/nvme0n1p3  1.5T  1.4T   13G 100% /
tmpfs           110G     0  110G   0% /dev/shm
tmpfs           5.0M     0  5.0M   0% /run/lock
tmpfs           110G     0  110G   0% /sys/fs/cgroup
/dev/loop0      128K  128K     0 100% /snap/bare/5
/dev/loop1       56M   56M     0 100% /snap/core18/2947
/dev/loop2      105M  105M     0 100% /snap/core/17247
/dev/loop3      105M  105M     0 100% /snap/core/17212
/dev/loop4       74M   74M     0 100% /snap/core22/2111
/dev/loop5       74M   74M     0 100% /snap/core22/2133
/dev/loop6       56M   56M     0 100% /snap/core18/2952
/dev/loop7       64M   64M     0 100% /snap/core20/2599
/dev/loop10      67M   67M     0 100% /snap/core24/1151
/dev/loop8       64M   64M     0 100% /snap/core20/2669
/dev/loop13     517M  517M     0 100% /snap/gnome-42-2204/202
/dev/loop21     4.7M  4.7M     0 100% /snap/rustup/1479
/dev/loop14      92M   92M     0 100% /snap/gtk-common-themes/1535
/dev/loop22      92M   92M     0 100% /snap/lxd/32662
/dev/loop23      92M   92M     0 100% /snap/lxd/29619
/dev/loop27      51M   51M     0 100% /snap/snapd/25202
/dev/loop16      15M   15M     0 100% /snap/kubeadm/3702
/dev/loop19      14M   14M     0 100% /snap/kubectl/3676
/dev/sdc        880G  180G  656G  22% /mnt/sdc
/dev/sdb1       2.0T  1.6T  304G  85% /home/homie/models
/dev/sdd        880G   28K  835G   1% /mnt/sdd
/dev/loop15      14M   14M     0 100% /snap/kubectl/3677
/dev/loop26      50M   50M     0 100% /snap/snapd/24792
/dev/loop18     517M  517M     0 100% /snap/gnome-42-2204/226
/dev/loop17      18M   18M     0 100% /snap/kubelet/3674
/dev/loop28      15M   15M     0 100% /snap/kubeadm/3703
/dev/loop20      15M   15M     0 100% /snap/kubelet/3677
/dev/loop24     2.8M  2.8M     0 100% /snap/mdbook/2
/dev/loop25     4.7M  4.7M     0 100% /snap/rustup/1492
/dev/nvme0n1p1  511M  6.1M  505M   2% /boot/efi
/dev/loop29      67M   67M     0 100% /snap/core24/1196
/dev/loop11     248M  248M     0 100% /snap/firefox/6933
/dev/loop9      248M  248M     0 100% /snap/firefox/6966
tmpfs            22G  184K   22G   1% /run/user/1000
tmpfs           2.0G  1.1G  977M  53% /home/gpu/dry/faiss/tutorial/cpp/memory_workspace

## 存储介质信息
- 普通磁盘: 本地文件系统
- 内存磁盘: tmpfs (2GB)

## 测试结果

### 普通磁盘测试结果


### 内存磁盘测试结果


## 性能对比分析

### 关键指标对比
| 指标 | 普通磁盘 | 内存磁盘 | 性能差异 |
|------|----------|----------|----------|
| QPS (20线程) | 3024.52 | 1741.67 | -42.00% |
| 延迟 (20线程) | 6.1614 ms | 10.0087 ms | -62.00% |

## 分析结论

### 磁盘I/O瓶颈分析
通过对比两种存储介质的性能表现，可以得出以下结论：

1. **高线程数下的性能差异**: 如果内存磁盘在高线程数下表现明显优于普通磁盘，说明存在磁盘I/O瓶颈
2. **最优线程数**: 找出两种存储介质下的最优线程配置
3. **性能提升幅度**: 量化磁盘I/O瓶颈对整体性能的影响

### 优化建议
- 如果存在明显性能差异，建议考虑使用SSD或增加内存缓存
- 根据测试结果调整线程数配置
- 考虑使用内存映射文件优化I/O性能

## 文件说明
- `disk_results.csv`: 普通磁盘测试结果
- `memory_results.csv`: 内存磁盘测试结果  
- `disk_test.log`: 普通磁盘测试日志
- `memory_test.log`: 内存磁盘测试日志
