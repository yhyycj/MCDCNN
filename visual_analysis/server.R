library(shiny)
library(ggplot2)
library(DT)

server<-function(input,output) {
    slt_ESN <- reactive({
        return(input$ESN)
    })
    slt_param <- reactive({
        return(input$Parameter)
    })
    
    data_plot <- reactive({
        ESN <- slt_ESN()
        param <- slt_param()
        return(df_records[df_records$tk_esn == ESN,])
    })
    
    output$plot <- renderPlot({
        ESN <- slt_ESN()
        param <- slt_param()
        data_plot <- data_plot()
        p <- ggplot(data = data_plot, aes(x = tk_flight_datetime,y = data_plot[,param])) + geom_point()
        onWing_date <- df_events[df_events$ESN == ESN, 'On.wing']
        event_date <- df_events[df_events$ESN == ESN, 'Off.wing']
        
        if (!is.na(event_date)) {
            p <- p + geom_vline(xintercept = as.numeric(event_date), color="red")
        }

        if (!is.na(onWing_date)) {
            p <- p + geom_vline(xintercept = as.numeric(onWing_date), color="blue")
        }
        
        p <- p + coord_cartesian(xlim = c(input$x_lim[1], input$x_lim[2]), ylim = c(input$y_lim[1], input$y_lim[2]))
        return(p)
    })
    
    output$table_event <- DT::renderDataTable({
        ESN <- slt_ESN()
        return(df_events[df_events$ESN == ESN,])
    }, rownames = NULL, options = list(scrollX = TRUE))
    
    output$table_records <- DT::renderDataTable({
        ESN <- slt_ESN()
        param <- slt_param()
        return(df_records[df_records$tk_esn == ESN,])  
    }, rownames = NULL, options = list(scrollX = TRUE))
    
    output$slider_x <- renderUI({
        data_plot <- data_plot()
        min_date <- min(data_plot$tk_flight_datetime, na.rm = TRUE)
        max_date <- max(data_plot$tk_flight_datetime, na.rm = TRUE)
        sliderInput("x_lim", "flight date", min = min_date, max = max_date, value = c(min_date, max_date))
    })
    
    output$slider_y <- renderUI({
        data_plot <- data_plot()
        param <- input$Parameter
        print(param)
        min_data <- min(data_plot[,param], na.rm = TRUE)
        max_data <- max(data_plot[,param], na.rm = TRUE)
        sliderInput("y_lim", "value range", min = min_data, max = max_data, value = c(min_data, max_data))
    })
}