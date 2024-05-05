/**
 * @file http_server.h
 * @author Lukáš Baštýř (l.bastyr@seznam.cz, 492875@muni.cz)
 * @brief http server with post support
 * @date 09-03-2024
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <map>
#include <algorithm>
#include <unordered_set>
#include <tuple>
#include <string>
#include "libs/base/http_server.h"
#include "libs/base/strings.h"


using namespace coralmicro;


//struct for storing uri and raw payload for specific connection
typedef struct {
    std::string uri;
    std::vector<uint8_t> payload;
} uri_payload_t;

//Http server with post support
class PostHttpServer : public HttpServer {
  private:
    std::map<void*, uri_payload_t> payload_buffer;  //payload buffers for storing connection payloads
    
    //post callback
    using PostUriHandler = std::function<std::string(std::string, std::vector<uint8_t>)>;
    PostUriHandler post_uri_handler;

  public:
    /** @brief Register a callback which will be called on POST finished with request URI path and payload.
     * 
     * @param handler POST handler callback
     * @return URI path of a file (must be handled by GET handler) to be returned as response body. Empty string otherwise
     */
    void AddPostUriHandler(PostUriHandler handler) {
        post_uri_handler = handler;
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

        //copy uri and reserver space
        payload_buffer[connection].uri = uri;
        payload_buffer[connection].payload.reserve(content_len);
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
        struct pbuf* current_pbuf = p;
        while (current_pbuf != nullptr) {
            uint8_t *payload = static_cast<uint8_t *>(current_pbuf->payload);
            payload_buffer[connection].payload.insert(payload_buffer[connection].payload.end(), &payload[0], &payload[current_pbuf->len]);

            current_pbuf = current_pbuf->next;
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
        //no matter what, filename just won't work. At least URI path does work
        //write a uri path to the response_uri of a file that is stored inside the device in its fs (/index.html for example is the main web page)
        //uri path must be handled by normal http server uri handler (http_server.AddUriHandler())
        //'/index.html' returns the main page
        //'/ok' returns 'ok.html' as urihandler returns it for '/ok' path

        //call callback
        std::string path = post_uri_handler(payload_buffer[connection].uri, payload_buffer[connection].payload);
        if (path.size())
            snprintf(response_uri, response_uri_len, "%s", path.c_str());

        //remove the now stale buffer
        payload_buffer.erase(connection);
    };
};
