<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                version="1.0">
    
    <xsl:template match="/" >
        
        <html>
            <head>
                <style>
                    .film-div {
                        display: block;
                        padding: 20px;
                        margin: 10px;
                        width: 100%;
                    }

                    .film-title-div {
                        width: 100%;
                        text-align: left;
                        color: rgb(9,9,121);
                        border-bottom: 3px solid rgb(9,9,121);
                        margin-bottom: 20px;
                    }

                    .film-title-div h2
                    {
                        margin: 0px;
                        padding: 0px;
                    }


                    .film-content-div
                    {
                        display: flex;
                    }

                    .film-left-div, .film-right-div
                    {
                        margin-right: 5%;
                    }

                    .film-left-div img{
                       height: 100%;
                    }
                
                </style>
            </head>
            <body>

                <xsl:apply-templates select="lord-of-the-rings"/>
                
                <xsl:for-each select="*/films/film">
                    <div class="film-div">
                        <div class="film-title-div">
                            <h2> <xsl:value-of select="title"/></h2>
                        </div>
                        <div class="film-content-div">
                            <div class="film-left-div">
                                <img src="{img-source/@value}" alt="{nombre}" width="150" height="150"/>
                            </div>
                            <div class="film-right-div">
                                <p> <b>Rélasiteur</b> : <xsl:value-of select="director"/> </p>
                                <p> <b>Date de sortie</b> : <xsl:value-of select="release-date"/> </p>
                                <p> <b>Box-office</b> : <xsl:value-of select="box-office"/> millions USD ()</p>
                                <p> <b>Roles principaux</b> :  </p>
                                <ul>

                                    <xsl:for-each select="roles/role">
                                        <li> 
                                            <i><xsl:value-of select="@character"/></i> - 
                                        </li>
                                    </xsl:for-each>
                                </ul>
                            </div>
                        </div>
                    </div>
                </xsl:for-each>
                
                
            </body>
        </html> 
    </xsl:template>
    <xsl:template match="lord-of-the-rings">
        <h1><xsl:value-of select="@name"/></h1>
    </xsl:template>
    
</xsl:stylesheet>