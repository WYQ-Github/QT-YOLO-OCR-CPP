#include "UdpTool.h"
#include <windows.h>
#define socklen_t int
UdpTool::UdpTool()
{
    std::cout << "Created UdpTool" << std::endl;
	//初始化动态链接库
	//引用lib库
	static bool first = true;
	if (first)
	{
		first = false;						 //只在首次进入时初始化网络库
		WSADATA ws;					         //加载Socket库   项目属性-链接器-输入加上 ws2_32.lib
		WSAStartup(MAKEWORD(2, 2), &ws);     //动态库引用加1
	}
}

UdpTool::~UdpTool()
{
    std::cout << "Destory UdpTool" << std::endl;
}

bool UdpTool::CreateSocket(UdpToolParam param, bool broadcast)						 //创建套接字
{
    m_udp_tool_param = param;
	uport = param.listen_port;
	usock = socket(AF_INET, SOCK_DGRAM, 0);
	//SOCKET usock = socket(AF_INET, SOCK_DGRAM, 0);  //TCP/IP  UDP  创建udp 套接字
	if (usock == -1)
	{
		printf("create udp socket failed.\n");
		return false;
	}
	printf("create udp socket success.\n");

	if(broadcast){
        const int opt = 1;
        int ret = setsockopt(usock, SOL_SOCKET, SO_BROADCAST, (char *)&opt, sizeof(opt));
        if(ret < 0)
        {
            printf("socket set broadcast failed\n");
            return false;
        }
        printf("socket set broadcast success.\n");
    }
    //SetRecvTimeout(m_udp_tool_param.)


	return true;
}

void UdpTool::Close()					  //关闭连接
{
	if (usock <= 0)
		return;  //socket出错
	closesocket(usock);		//已宏定义

	usock = 0;
	uport = 0;
}

bool UdpTool::Bind()			 //绑定并监听端口号
{
	sockaddr_in saddr;              //数据结构
	saddr.sin_family = AF_INET;     //协议
	saddr.sin_port = htons(uport);   //端口，主机字节序（小端方式）转换成网络字节序（大端方式）
									 //  saddr.sin_addr.S_un.S_addr = htonl(INADDR_ANY);
	saddr.sin_addr.s_addr = htonl(INADDR_ANY);   //绑定IP到广播地址INADDR_ANY 0.0.0.0  为了兼容linux

	if (bind(usock, (sockaddr*)&saddr, sizeof(saddr)) != 0)   //安装sockaddr_in数据结构绑定套接字
	{
		printf("udp bind port %d failed.\n", uport);
		return false;
	}
	printf("udp bind port %d success.\n", uport);
	return true;
}

int UdpTool::Send(const char* buf, int size, char* ip, unsigned short port)	 //发送数据(强制全部发送)
{
	int sendedSize = 0;   //已发送成功的长度
	sockaddr_in saddr;
	saddr.sin_family = AF_INET;  //tcp/ip协议
	saddr.sin_port = htons(port);//服务端的端口  主机字节序转换成网络字节序
	saddr.sin_addr.s_addr = inet_addr(ip);  //本机的ip地址  字符串ip地址转成整型
	while (sendedSize != size)   //若没发送完成，则从断点开始继续发送 直到完成
	{
		int len = sendto(usock, buf + sendedSize, size - sendedSize, 0, (sockaddr*)&saddr, sizeof(saddr));
		if (len <= 0)break;
		sendedSize += len;
	}
	return sendedSize;
}


int UdpTool::Recv(char* buf, int bufsize, sockaddr_in* from)	 //接收数据
{
	socklen_t len = sizeof(sockaddr_in);
	int re = recvfrom(usock, buf, bufsize, 0, (sockaddr*)from, &len);   //返回接收的客户端的网络地址，存在在地址中
	return re;
}

int UdpTool::SetRecvTimeout(int sec = 1)   //设置udp接收超时
{
	int udp_rev_time = sec * 1000;
	if (setsockopt(usock, SOL_SOCKET, SO_RCVTIMEO, (char*)&udp_rev_time, sizeof(int)) < 0)
	{
		printf("set udp receive failed.\n");
		return -1;
	}
	printf("set udp recv timeout success. %d seconds\n", sec);
	return 0;
}


int UdpTool::SetSendTimeout(int sec = 1)   //设置udp发送超时
{
	int udp_rev_time = sec;
	if (setsockopt(usock, SOL_SOCKET, SO_SNDTIMEO, (char*)&udp_rev_time, sizeof(int)) < 0)
	{
		printf("set udp send failed.");
		return -1;
	}
	return 0;
	printf("set udp send timeout success. %d seconds\n", sec);
}
