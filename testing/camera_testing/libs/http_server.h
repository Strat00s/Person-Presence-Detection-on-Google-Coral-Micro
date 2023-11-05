#pragma once

#include <map>
#include <algorithm>
#include <unordered_set>
#include <tuple>
#include "libs/base/http_server.h"
#include "libs/base/strings.h"


using namespace coralmicro;


//struct for storing uri and raw payload
typedef struct {
    std::string uri;
    std::vector<uint8_t> payload;
} payload_uri_t;

//Http server with post support
class PostHttpServer : public HttpServer {
  private:
    std::map<void*, payload_uri_t> payload_buffer;              //payload buffers while processing payloads
    std::unordered_set<std::string> post_paths;                 //supported post paths
    void (*postFinishedCallback)(payload_uri_t payload_uri);    //function to call once all data are saved

  public:

    /** @brief Register a callback function which should be executed once POST is finished
     * 
     * @param func Function to call with argument of type payload_uri_t
     */
    void registerPostFinishedCallback(void (*func)(payload_uri_t)) {
        postFinishedCallback = func;
    }

    /** @brief Add post path for which post are accepted
     * 
     * @param post_path supported post path
     */
    void addPostPath(std::string post_path) {
        post_paths.insert(post_path);
    }

    // Called when an HTTP POST request is first received.
    // This must be implemented by subclasses that want to handle posts.
    //
    // @param connection Unique connection identifier, valid until
    // `PostFinished()` is called.
    // @param uri The HTTP header URI receiving the POST request.
    // @param http_request  The raw HTTP request (the first packet, normally).
    // @param http_request_len  Size of 'http_request'.
    // @param content_len Content-Length from HTTP header.
    // @param response_uri Filename of response file, to be filled when denying
    //   the request.
    // @param response_uri_len Size of the 'response_uri' buffer.
    // @param post_auto_wnd Set this to 0 to let the callback code handle window
    //   updates by calling `httpd_post_data_recved` (to throttle rx speed)
    //   default is 1 (httpd handles window updates automatically)
    err_t PostBegin(void* connection, const char* uri, const char* http_request, u16_t http_request_len,
                    int content_len, char* response_uri, u16_t response_uri_len, u8_t* post_auto_wnd) {
        (void)http_request;
        (void)http_request_len;
        (void)response_uri;
        (void)response_uri_len;
        (void)post_auto_wnd;

        //post_paths.insert(uri);
        if (post_paths.find(uri) == post_paths.end()) {
            printf("unsuported post path\r\n");
            return ERR_ARG;
        }

        //copy uri and reserver space
        payload_buffer[connection].uri = uri;
        payload_buffer[connection].payload.reserve(content_len);
        printf("Everything ok\r\n");
        return ERR_OK;
    }

    // Called for each packet buffer of data that is received for a POST.
    //
    // @param connection  Unique connection identifier.
    // @param p  Received data as a
    // [pbuf](https://www.nongnu.org/lwip/2_1_x/structpbuf.html).
    // **ATTENTION:** Your application is responsible for freeing the pbufs!
    err_t PostReceiveData(void* connection, struct pbuf* p) {
        (void)connection;
        
        //go through buffers and copy them to main buffer
        struct pbuf* cp = p;
        while (cp != nullptr) {
            uint8_t *byte_payload = static_cast<uint8_t *>(cp->payload);
            for (int i = 0; i < cp->len; i++)
                payload_buffer[connection].payload.push_back(byte_payload[i]);

            cp = p->next;
        }

        pbuf_free(p);

        return ERR_OK;
    };

    // Called when all data is received or when the connection is closed. The
    // application must return the filename/URI of a file to send in response to
    // this POST request. If the response_uri buffer is untouched, a 404 response
    // is returned.
    //
    // @param connection Unique connection identifier.
    // @param response_uri Filename of response file, to be filled when denying
    //   the request.
    // @param response_uri_len Size of the 'response_uri' buffer.
    void PostFinished(void* connection, char* response_uri, u16_t response_uri_len) {
        //(void)connection;
        //TODO fill response uri
        (void)response_uri;
        (void)response_uri_len;
        
        //call callback
        postFinishedCallback(payload_buffer[connection]);
        
        //remove current buffer
        payload_buffer.erase(connection);
    };
};
