ui <- fluidPage(
    sidebarLayout(
        sidebarPanel(
            fluidRow(
                column(12, selectInput("ESN", "ESN", choices = df_events$ESN)),
                column(12, selectInput("Parameter", "Parameter", choices = df_params[df_params[2] == 1, 1])),
                column(12, uiOutput("slider_x")),
                column(12, uiOutput("slider_y"))
            )
        ),
        mainPanel(
            fluidRow(plotOutput("plot")),
            fluidRow(div(dataTableOutput("table_event"), style = "font-size:75%")),
            fluidRow(div(dataTableOutput("table_records"), style = "font-size:75%"))
        )
    )
)