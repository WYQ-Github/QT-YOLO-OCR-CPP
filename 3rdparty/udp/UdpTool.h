#pragma once
#define NOMINMAX  // 放在包含 Windows.h 之前
#include <algorithm>
#include <iostream>
#include <string>
#include <windows.h>
#define socklen_t int
#include <cstring>


struct UdpToolParam
{
    char listen_ip[32];
    char send_ip[32];
    int listen_port;
    int send_port;

    ~UdpToolParam()
    {

    };
    UdpToolParam(){
		
    }

    void operator=(const UdpToolParam& other)
	{
		strncpy_s(listen_ip, other.listen_ip, sizeof(listen_ip));
		strncpy_s(send_ip, other.send_ip, sizeof(send_ip));
        listen_port = other.listen_port;
        send_port = other.send_port;
	}
};

class UdpTool
{
public:
	UdpToolParam m_udp_tool_param;
	bool CreateSocket(UdpToolParam param, bool broadcast);						 //创建套接字
	void Close();						     //关闭连接
	bool Bind();							 //绑定并监听端口号
	int Send(const char* buf, int size, char* ip, unsigned short port);	 //发送数据
	int Recv(char* buf, int bufsize, sockaddr_in* from);				//接收数据

	int SetRecvTimeout(int sec);			 //设置udp接收超时
	int SetSendTimeout(int sec);		     //设置udp发送超时

	UdpTool();
	virtual ~UdpTool();

private:
	int usock = 0;  //udp服务端的socket create成员函数自己生成
	unsigned short uport = 0;   //构造函数从外获取
};

