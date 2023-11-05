#include "apps/camera_testing/libs/http_server.h"

err_t PostBegin(void* connection, const char* uri, const char* http_request, u16_t http_request_len,
                int content_len, char* response_uri, u16_t response_uri_len, u8_t* post_auto_wnd) {
            (void)connection;
        (void)uri;
        (void)http_request;
        (void)http_request_len;
        (void)content_len;
        (void)response_uri;
        (void)response_uri_len;
        (void)post_auto_wnd;
        return ERR_ARG;
}